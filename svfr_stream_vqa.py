import os
import torch
import gc
from logzero import logger

from base import BaseVQA, work
from video_loader import stream_video_blocks

class SVFRStreamVQA(BaseVQA):
    """
    Implements the core logic for Sequential Visual Feature Refinement (SVFR).
    This version includes a robust, three-step memory cleanup to prevent leaks.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieve_per_chunk = self.args.retrieve_per_chunk
        logger.info(f"Initialized SVFRStreamVQA with retrieve_per_chunk={self.retrieve_per_chunk}")

    def analyze_a_video(self, video_sample):
        video_id = str(video_sample.get("video_id"))
        
        video_filename = video_id if video_id.endswith('.mp4') else video_id + '.mp4'
        video_path = os.path.join(self.args.video_dir_path, video_filename)
        
        conversations = video_sample.get("conversations", [])

        if not os.path.exists(video_path):
            logger.warning(f"[{video_id}] Video path not found; skipping: {video_path}")
            self.write_csv_row(video_id=video_id, question="N/A", pred_answer="Skipped: Video not found")
            return
        
        for i, conv in enumerate(conversations):
            question = conv.get("question")
            if not question:
                logger.warning(f"[{video_id}] Skipping conversation {i+1} due to missing question.")
                continue

            logger.info(f"[{video_id}] Processing Q{i+1}/{len(conversations)}: '{question[:80]}...'")
            
            self.memory_bank.clear()
            
            num_blocks = 0
            try:
                for block_frames in stream_video_blocks(video_path, self.sample_fps, self.block_size):
                    num_blocks += 1
                    logger.debug(f"[{video_id}] Processing Block {num_blocks} with {len(block_frames)} frames.")
                    
                    try:
                        retrieved_feats = self.qa_model.retrieve_features_from_chunk(
                            frames=block_frames,
                            question=question,
                            retrieve_size=self.retrieve_per_chunk
                        )
                        
                        if retrieved_feats is not None and retrieved_feats.shape[0] > 0:
                            self.memory_bank.add(retrieved_feats)
                            logger.debug(f"Added {retrieved_feats.shape[0]} features. Memory bank size: {len(self.memory_bank)}")
                        
                        del retrieved_feats # Explicitly delete tensor after use

                    except Exception as e:
                        logger.error(f"[{video_id}] Feature retrieval failed for Block {num_blocks}: {e}", exc_info=True)
                        continue
            except IOError as e:
                 logger.error(f"[{video_id}] Failed to stream video frames: {e}")
                 self.write_csv_row(video_id=video_id, question=question, pred_answer=f"Error: Failed to stream video")
                 continue

            visual_memory = self.memory_bank.get_all()
            if visual_memory.shape[0] == 0:
                logger.warning(f"[{video_id}] Visual memory is empty for question '{question}'.")
            
            logger.info(f"[{video_id}] Accumulated {visual_memory.shape[0]} total features for final answer generation.")
            
            pred_answer = ""
            try:
                pred_answer = self.qa_model.answer_question(question, visual_memory)
            except Exception as e:
                pred_answer = f"ERROR: {e}"
                logger.error(f"[{video_id}] Final QA generation failed for Q: '{question}' | {e}", exc_info=True)
            
            logger.info(f"[{video_id}] Q: {question} | A: {pred_answer}")
            self.write_csv_row(
                video_id=video_id,
                question=question,
                gt_answer=conv.get("answer", ""),
                pred_answer=pred_answer
            )
            
            # *** THE DEFINITIVE MEMORY LEAK FIX ***
            del visual_memory  # 1. Explicitly delete the large tensor
            gc.collect()       # 2. Trigger Python's garbage collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 3. Clear PyTorch's cache

if __name__ == "__main__":
    work(SVFRStreamVQA)

