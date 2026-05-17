import torch
from logzero import logger

def load_model(model_name_or_path: str, model_type: str, device: str, torch_dtype):
    """
    Loads a model and its processor based on the specified model_type.
    This function acts as a factory for different model architectures.
    Returns: (model, processor)
    """
    logger.info(f"Attempting to load model '{model_name_or_path}' of type '{model_type}'")
    
    try:
        if model_type == "llava_onevision":
            from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor
            model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            processor = LlavaOnevisionProcessor.from_pretrained(model_name_or_path)
            model.eval()
            return model, processor

        elif model_type == "video_llava":
            from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
            model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
            ).to(device)
            processor = VideoLlavaProcessor.from_pretrained(model_name_or_path)
            model.eval()
            return model, processor

        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported in this loader.")

    except Exception as e:
        logger.error(f"Failed to load model '{model_name_or_path}'. Please ensure the path is correct and dependencies are installed.")
        raise RuntimeError(f"Model loading error: {e}")
