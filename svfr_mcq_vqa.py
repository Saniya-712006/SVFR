import os
import torch
import gc
from logzero import logger

# -------------------------new current latency calculation code modification ---------------
import time
from decord import VideoReader, cpu
# -------------------------- end of the modification --------------------------

from base import BaseVQA, work
from video_loader import stream_video_blocks

class SVFRMCQVQA(BaseVQA):
    """
    Implements SVFR for Multiple-Choice Question Answering.
    This version uses forced-choice generation instead of independent scoring.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.retrieve_per_chunk = self.args.retrieve_per_chunk
        logger.info(f"Initialized SVFRMCQVQA with retrieve_per_chunk={self.retrieve_per_chunk}")

    def analyze_a_video(self, video_sample):
        """
        Processes a single video for MCQ using SVFR + forced choice.
        """
        video_id = str(video_sample.get("video_id"))
        video_filename = video_id if video_id.endswith('.mp4') else video_id + '.mp4'
        video_path = os.path.join(self.args.video_dir_path, video_filename)
        
        conversations = video_sample.get("conversations", [])

        if not os.path.exists(video_path):
            logger.warning(f"[{video_id}] Video path not found; skipping: {video_path}")
            for conv in conversations:
                self.write_csv_row(video_id=video_id, question=conv.get("question", "N/A"), pred_answer="Skipped: Video not found")
            return
        
        # -------------------- latency calculation modification ----------------------
        video_duration = 0.0
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            video_duration = len(vr) / vr.get_avg_fps()
        except Exception as e:
            logger.warning(f"[{video_id}] Could not determine video duration: {e}")
        
        # -------------------- end of modification ----------------------
        for i, conv in enumerate(conversations):
            question = conv.get("question")
            choices = conv.get("choices")
            gt_answer = conv.get("answer")

            if not question or not choices:
                logger.warning(f"[{video_id}] Skipping Q{i+1} due to missing question or choices.")
                continue

            logger.info(f"[{video_id}] Processing Q{i+1}/{len(conversations)}: '{question[:80]}...'")
            
            # -------------------- latency calculation modification ----------------------
            start_time = time.time() # Start timer
            # -------------------- end of modification ----------------------
            self.memory_bank.clear()
            num_blocks = 0
            try:
                for block_frames in stream_video_blocks(video_path, self.sample_fps, self.block_size):
                    num_blocks += 1
                    try:
                        retrieved_feats = self.qa_model.retrieve_features_from_chunk(
                            frames=block_frames,
                            question=question,
                            retrieve_size=self.retrieve_per_chunk
                        )
                        if retrieved_feats is not None and retrieved_feats.shape[0] > 0:
                            self.memory_bank.add(retrieved_feats)
                        del retrieved_feats 
                    except Exception as e:
                        logger.error(f"[{video_id}] Feature retrieval failed for block {num_blocks}: {e}", exc_info=True)
            except IOError as e:
                 logger.error(f"[{video_id}] Failed to stream video frames: {e}")
                 self.write_csv_row(video_id=video_id, question=question, gt_answer=gt_answer, pred_answer=f"Error: Video stream failed")
                 continue
            
            visual_memory = self.memory_bank.get_all()
            logger.info(f"[{video_id}] Accumulated {visual_memory.shape[0]} total features.")

            # NEW: Use forced choice instead of scoring each choice independently
            try:
                predicted_answer = self.qa_model.choose_best_answer(
                    question=question,
                    visual_memory=visual_memory,
                    choices=choices
                )
            except Exception as e:
                logger.error(f"[{video_id}] Forced choice failed: {e}", exc_info=True)
                predicted_answer = f"Error: {str(e)}"

            # -------------------- latency calculation modification ----------------------
            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            # -------------------- end of modification ----------------------
            
            logger.info(f"[{video_id}] Q: {question} | Pred: {predicted_answer} | GT: {gt_answer}")
            self.write_csv_row(
                video_id=video_id,
                question=question,
                gt_answer=gt_answer,
                pred_answer=predicted_answer,

                # -------------------- latency calculation modification ----------------------
                video_duration_s=round(video_duration, 2),
                inference_time_ms=round(inference_time_ms, 2)
                # -------------------- end of modification ----------------------
            )

            del visual_memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    work(SVFRMCQVQA)