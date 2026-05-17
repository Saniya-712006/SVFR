import torch
import torch.nn.functional as F
from PIL import Image
from logzero import logger
from typing import List
import re

# Correct: Only import what is needed from model_loader
from model_loader import load_model

# Correct: Define MODEL_MAP here, where it is used.
MODEL_MAP = {
    "llava_onevision_0.5b": {
        "path": "model_zoo/llava-onevision-qwen2-0.5b-ov-hf", "type": "llava_onevision"
    },
    "llava_onevision_7b": {
        "path": "model_zoo/llava-onevision-qwen2-7b-ov-hf", "type": "llava_onevision"
    },
    "video_llava_7b": {
        "path": "model_zoo/Video-LLaVA-7B-hf", "type": "video_llava"
    }
}

class VideoLLMWrapper:
    def __init__(self, model_path, model_type, args, dtype=torch.float16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.args = args
        self.model, self.processor = load_model(
            model_path, model_type=model_type, device=self.device, torch_dtype=dtype
        )
        self.tokenizer = self.processor.tokenizer
        logger.info(f"Model and processor loaded to {self.device}")

    @torch.no_grad()
    def encode_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        if not frames: return torch.empty(0, device=self.device)
        try:
            inputs = self.processor(text=[""], images=frames, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device, dtype=self.dtype)
            b, t, c, h, w = pixel_values.shape
            pixel_values_reshaped = pixel_values.view(b * t, c, h, w)
            vision_tower = self.model.vision_tower
            vision_outputs = vision_tower(pixel_values_reshaped, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            return image_features[:, 0, :].to(torch.float32)
        except Exception as e:
            logger.error(f"Frame encoding failed: {e}", exc_info=True)
            return torch.empty(0, device=self.device)

    @torch.no_grad()
    def retrieve_features_from_chunk(self, frames: List[Image.Image], question: str, retrieve_size: int) -> torch.Tensor:
        visual_features = self.encode_frames(frames)
        if visual_features.shape[0] == 0: return torch.empty(0, device=self.device)
        inputs = self.tokenizer(question, return_tensors="pt").to(self.device)
        text_embeddings = self.model.get_input_embeddings()(inputs.input_ids)
        text_features = text_embeddings.mean(dim=1)
        projected_visual_features = self.model.multi_modal_projector(visual_features.to(self.dtype))
        text_features_norm = F.normalize(text_features, p=2, dim=-1)
        visual_features_norm = F.normalize(projected_visual_features, p=2, dim=-1)
        similarity_scores = text_features_norm @ visual_features_norm.T
        k = min(retrieve_size, visual_features.shape[0])
        _, top_indices = torch.topk(similarity_scores.squeeze(0), k=k)
        return visual_features[top_indices]

    @torch.no_grad()
    def score_answer_choice(self, question: str, visual_memory: torch.Tensor, choice_text: str) -> float:
        if visual_memory is None or visual_memory.nelement() == 0: return -float('inf')
        prompt_pre = f"USER: <image>\n{question}\nASSISTANT: "
        prompt_post = choice_text
        pre_ids = self.tokenizer(prompt_pre, return_tensors="pt").input_ids.to(self.device)
        post_ids = self.tokenizer(prompt_post, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        text_embeds_layer = self.model.get_input_embeddings()
        pre_embeds = text_embeds_layer(pre_ids)
        post_embeds = text_embeds_layer(post_ids)
        projected_visual = self.model.multi_modal_projector(visual_memory.to(self.dtype)).to(pre_embeds.dtype).unsqueeze(0)
        
        # Manually find the image token placeholder index
        image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
        image_token_index = (pre_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(image_token_index) == 0: raise ValueError("Image token not found in prompt")
        image_token_index = image_token_index[0]

        inputs_embeds = torch.cat([pre_embeds[:, :image_token_index], projected_visual, pre_embeds[:, image_token_index+1:], post_embeds], dim=1)
        attention_mask = torch.ones_like(inputs_embeds[..., 0])
        
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs.logits
        
        target_logits = logits[0, -post_ids.shape[1]:, :]
        log_probs = F.log_softmax(target_logits, dim=-1)
        target_log_probs = log_probs.gather(dim=1, index=post_ids.squeeze(0).unsqueeze(-1)).squeeze(-1)
        return target_log_probs.mean().item()

    @torch.no_grad()
    def choose_best_answer(self, question: str, visual_memory: torch.Tensor, choices: List[str]) -> str:
        """
        NEW METHOD: Forced-choice generation.
        Present all choices to the model at once and ask it to select the best one.
        """
        if visual_memory is None or visual_memory.nelement() == 0:
            return choices[0] if choices else "Error: No visual memory"
        
        try:
            # Build prompt with all choices
            prompt = f"USER: <image>\nQuestion: {question}\n\nChoices:\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\n"
            prompt += "\nASSISTANT: The correct answer is"
            
            # Tokenize
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # Find image token
            image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
            image_token_index = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0][0]

            # Get embeddings
            text_embeds = self.model.get_input_embeddings()(input_ids)
            pre_embeds = text_embeds[:, :image_token_index]
            post_embeds = text_embeds[:, image_token_index + 1:]

            # Project visual memory
            projected_visual = self.model.multi_modal_projector(visual_memory.to(self.dtype)).to(text_embeds.dtype).unsqueeze(0)
            
            # Construct input
            final_embeds = torch.cat([pre_embeds, projected_visual, post_embeds], dim=1)
            attention_mask = torch.ones_like(final_embeds[..., 0])

            # Generate
            output_ids = self.model.generate(
                inputs_embeds=final_embeds, 
                attention_mask=attention_mask, 
                max_new_tokens=64,
                do_sample=False
            )
            
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            generated_text = response.split("ASSISTANT:")[-1].strip()
            
            logger.debug(f"Generated: {generated_text}")
            
            # Parse answer
            return self._extract_choice(generated_text, choices)

        except Exception as e:
            logger.error(f"Forced choice failed: {e}", exc_info=True)
            return choices[0] if choices else "Error"
    
    def _extract_choice(self, generated_text: str, choices: List[str]) -> str:
        """Extract the choice from generated text"""
        gen_upper = generated_text.upper().strip()
        
        # Try letter match (A, B, C, D, E)
        letter_match = re.search(r'\b([A-E])\b', gen_upper)
        if letter_match:
            idx = ord(letter_match.group(1)) - ord('A')
            if 0 <= idx < len(choices):
                return choices[idx]
        
        # Try exact text match
        for choice in choices:
            if choice.lower() in generated_text.lower():
                return choice
        
        # Fallback
        logger.warning(f"Could not parse: '{generated_text}'")
        return choices[0]

    @torch.no_grad()
    def answer_question(self, question: str, visual_memory: torch.Tensor) -> str:
        if visual_memory is None or visual_memory.nelement() == 0:
            prompt = f"USER: {question}\nASSISTANT:"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(**inputs, max_new_tokens=128)
            return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        try:
            prompt = f"USER: <image>\n{question} ASSISTANT:"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            image_token_id = self.tokenizer.convert_tokens_to_ids('<image>')
            image_token_index = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0][0]

            text_embeds = self.model.get_input_embeddings()(input_ids)
            pre_embeds = text_embeds[:, :image_token_index]
            post_embeds = text_embeds[:, image_token_index + 1:]

            projected_visual = self.model.multi_modal_projector(visual_memory.to(self.dtype)).to(text_embeds.dtype).unsqueeze(0)
            
            final_embeds = torch.cat([pre_embeds, projected_visual, post_embeds], dim=1)
            attention_mask = torch.ones_like(final_embeds[..., 0])

            output_ids = self.model.generate(inputs_embeds=final_embeds, attention_mask=attention_mask, max_new_tokens=128)
            response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            return response.split("ASSISTANT:")[-1].strip()

        except Exception as e:
            logger.error(f"Answer generation failed: {e}", exc_info=True)
            return f"Error: {e}"