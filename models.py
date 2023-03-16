import cv2
import requests
import openai
from scipy.constants import R
import torch
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image
from transformers import (
    AutoImageProcessor,
    BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    BlipProcessor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
    UperNetForSemanticSegmentation,
    pipeline,
)

from utils import *


class DALLE:
    def __init__(self, device):
        print("Generating image using DALLE")
        self.a_prompt = "best quality, extremely detailed"

    @prompts(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, inputs):
        # image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        # image = Image.open(image_path)
        instruct_text = inputs
        prompt = instruct_text + ", " + self.a_prompt
        # image = self.pipe(prompt, miage, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
        #                   guidance_scale=9.0).images[0]
        # updated_image_path = get_new_image_name(image_path, func_name="line2image")
        # image.save(updated_image_path)
        # print(f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
        #       f"Output Text: {updated_image_path}")

        response = openai.Image.create(prompt=prompt, n=1, size="1024x1024")
        image_url = response["data"][0]["url"]
        r = requests.get(image_url, stream=True)
        # updated_image_path = get_new_image_name(image_path, func_name="dalle")
        updated_image_path = os.path.join("image", str(uuid.uuid4())[0:8] + ".png")
        if r.status_code == 200:
            with open(updated_image_path, "wb") as f:
                f.write(r.content)
        print(
            f"\nProcessed DALLE, Input Text: {instruct_text}, "
            f"Output Text: {updated_image_path}"
        )
        return updated_image_path


class DALLE_EDITING:
    def __init__(self, device):
        print("Generating image using DALLE")
        self.a_prompt = "best quality, extremely detailed"

    @prompts(
        name="Remove Something From The Photo",
        description="useful when you want to remove an object or something from the photo "
        "from its description or location. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @prompts(
        name="Replace Something From The Photo",
        description="useful when you want to replace an object from the object description or "
        "location with another object from its description. "
        "The input to this tool should be a comma seperated string of three, "
        "representing the image_path, the object to be replaced, the object to be replaced with ",
    )
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(
            prompt=replace_with_txt,
            image=original_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
        ).images[0]
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Whisper:
    def __init__(self, device):
        print("Transcribing Audio by Whisper")

    @prompts(
        name="Get Audio Transcription",
        description="useful when you want to know what is inside the audio recording. receives audio_path as input. "
        "The input to this tool should be a string, representing the image_path. ",
    )
    def inference(self, audio_path):
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            print(
                f"\nProcessed SpeechRecognition, Input Image: {audio_path}, Output Text: {transcript}"
            )
            return transcript


class MaskFormer:
    def __init__(self, device):
        print("Initializing MaskFormer to %s" % device)
        self.device = device
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined"
        ).to(device)

    def inference(self, image_path, text):
        threshold = 0.5
        min_area = 0.02
        padding = 20
        original_image = Image.open(image_path)
        image = original_image.resize((512, 512))
        inputs = self.processor(
            text=text, images=image, padding="max_length", return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        if area_ratio < min_area:
            return None
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(
                slice(max(0, i - padding), i + padding + 1) for i in idx
            )
            mask_array[padded_slice] = True
        visual_mask = (mask_array * 255).astype(np.uint8)
        image_mask = Image.fromarray(visual_mask)
        return image_mask.resize(original_image.size)


class ImageEditing:
    def __init__(self, device):
        print("Initializing ImageEditing to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device)
        self.revision = "fp16" if "cuda" in device else None
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        ).to(device)

    @prompts(
        name="Remove Something From The Photo",
        description="useful when you want to remove an object or something from the photo "
        "from its description or location. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the object need to be removed. ",
    )
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @prompts(
        name="Replace Something From The Photo",
        description="useful when you want to replace an object from the object description or "
        "location with another object from its description. "
        "The input to this tool should be a comma seperated string of three, "
        "representing the image_path, the object to be replaced, the object to be replaced with ",
    )
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(
            prompt=replace_with_txt,
            image=original_image.resize((512, 512)),
            mask_image=mask_image.resize((512, 512)),
        ).images[0]
        updated_image_path = get_new_image_name(
            image_path, func_name="replace-something"
        )
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class InstructPix2Pix:
    def __init__(self, device):
        print("Initializing InstructPix2Pix to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

    @prompts(
        name="Instruct Image Using Text",
        description="useful when you want to the style of the image to be like the text. "
        "like: make it look like a painting. or make it like a robot. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the text. ",
    )
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ",".join(inputs.split(",")[1:])
        original_image = Image.open(image_path)
        image = self.pipe(
            text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        print(
            f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Text2Image:
    def __init__(self, device):
        print("Initializing Text2Image to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=self.torch_dtype
        )
        self.pipe.to(device)
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image From User Input Text",
        description="useful when you want to generate an image from a user input text and save it to a file. "
        "like: generate an image of an object or something, or generate an image that includes some objects. "
        "The input to this tool should be a string, representing the text used to generate image. ",
    )
    def inference(self, text):
        image_filename = os.path.join("image", str(uuid.uuid4())[0:8] + ".png")
        prompt = text + ", " + self.a_prompt
        image = self.pipe(prompt, negative_prompt=self.n_prompt).images[0]
        image.save(image_filename)
        print(
            f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}"
        )
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print("Initializing ImageCaptioning to %s" % device)
        self.device = device
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @prompts(
        name="Get Photo Description",
        description="useful when you want to know what is inside the photo. receives image_path as input. "
        "The input to this tool should be a string, representing the image_path. ",
    )
    def inference(self, image_path):
        inputs = self.processor(Image.open(image_path), return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        captions = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}"
        )
        return captions


class Image2Canny:
    def __init__(self, device):
        print("Initializing Image2Canny")
        self.low_threshold = 100
        self.high_threshold = 200

    @prompts(
        name="Edge Detection On Image",
        description="useful when you want to detect the edge of the image. "
        "like: detect the edges of this image, or canny detection on image, "
        "or perform edge detection on this image, or detect the canny image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        image = np.array(image)
        canny = cv2.Canny(image, self.low_threshold, self.high_threshold)
        canny = canny[:, :, None]
        canny = np.concatenate([canny, canny, canny], axis=2)
        canny = Image.fromarray(canny)
        updated_image_path = get_new_image_name(inputs, func_name="edge")
        canny.save(updated_image_path)
        print(
            f"\nProcessed Image2Canny, Input Image: {inputs}, Output Text: {updated_image_path}"
        )
        return updated_image_path


class CannyText2Image:
    def __init__(self, device):
        print("Initializing CannyText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-canny",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Canny Image",
        description="useful when you want to generate a new real image from both the user desciption and a canny image."
        " like: generate a real image of a object or something from this canny image,"
        " or generate a new real image of a object or something from this edge image. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description. ",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="canny2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed CannyText2Image, Input Canny: {image_path}, Input Text: {instruct_text}, "
            f"Output Text: {updated_image_path}"
        )
        return updated_image_path


class Image2Line:
    def __init__(self, device):
        print("Initializing Image2Line")
        self.detector = MLSDdetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Line Detection On Image",
        description="useful when you want to detect the straight line of the image. "
        "like: detect the straight lines of this image, or straight line detection on image, "
        "or peform straight line detection on this image, or detect the straight line image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        mlsd = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="line-of")
        mlsd.save(updated_image_path)
        print(
            f"\nProcessed Image2Line, Input Image: {inputs}, Output Line: {updated_image_path}"
        )
        return updated_image_path


class LineText2Image:
    def __init__(self, device):
        print("Initializing LineText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-mlsd", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Line Image",
        description="useful when you want to generate a new real image from both the user desciption "
        "and a straight line image. "
        "like: generate a real image of a object or something from this straight line image, "
        "or generate a new real image of a object or something from this straight lines. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description. ",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="line2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed LineText2Image, Input Line: {image_path}, Input Text: {instruct_text}, "
            f"Output Text: {updated_image_path}"
        )
        return updated_image_path


class Image2Hed:
    def __init__(self, device):
        print("Initializing Image2Hed")
        self.detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Hed Detection On Image",
        description="useful when you want to detect the soft hed boundary of the image. "
        "like: detect the soft hed boundary of this image, or hed boundary detection on image, "
        "or peform hed boundary detection on this image, or detect soft hed boundary image of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        hed = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="hed-boundary")
        hed.save(updated_image_path)
        print(
            f"\nProcessed Image2Hed, Input Image: {inputs}, Output Hed: {updated_image_path}"
        )
        return updated_image_path


class HedText2Image:
    def __init__(self, device):
        print("Initializing HedText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-hed", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Soft Hed Boundary Image",
        description="useful when you want to generate a new real image from both the user desciption "
        "and a soft hed boundary image. "
        "like: generate a real image of a object or something from this soft hed boundary image, "
        "or generate a new real image of a object or something from this hed boundary. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="hed2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed HedText2Image, Input Hed: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Image2Scribble:
    def __init__(self, device):
        print("Initializing Image2Scribble")
        self.detector = HEDdetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Sketch Detection On Image",
        description="useful when you want to generate a scribble of the image. "
        "like: generate a scribble of this image, or generate a sketch from this image, "
        "detect the sketch from this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        scribble = self.detector(image, scribble=True)
        updated_image_path = get_new_image_name(inputs, func_name="scribble")
        scribble.save(updated_image_path)
        print(
            f"\nProcessed Image2Scribble, Input Image: {inputs}, Output Scribble: {updated_image_path}"
        )
        return updated_image_path


class ScribbleText2Image:
    def __init__(self, device):
        print("Initializing ScribbleText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-scribble",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Sketch Image",
        description="useful when you want to generate a new real image from both the user desciption and "
        "a scribble image or a sketch image. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="scribble2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed ScribbleText2Image, Input Scribble: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Image2Pose:
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    @prompts(
        name="Pose Detection On Image",
        description="useful when you want to detect the human pose of the image. "
        "like: generate human poses of this image, or generate a pose image from this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        pose = self.detector(image)
        updated_image_path = get_new_image_name(inputs, func_name="human-pose")
        pose.save(updated_image_path)
        print(
            f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}"
        )
        return updated_image_path


class PoseText2Image:
    def __init__(self, device):
        print("Initializing PoseText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.num_inference_steps = 20
        self.seed = -1
        self.unconditional_guidance_scale = 9.0
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Pose Image",
        description="useful when you want to generate a new real image from both the user desciption "
        "and a human pose image. "
        "like: generate a real image of a human from this human pose image, "
        "or generate a new real image of a human from this pose. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pose2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed PoseText2Image, Input Pose: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Image2Seg:
    def __init__(self, device):
        print("Initializing Image2Seg")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.ade_palette = [
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
            [4, 250, 7],
            [224, 5, 255],
            [235, 255, 7],
            [150, 5, 61],
            [120, 120, 70],
            [8, 255, 51],
            [255, 6, 82],
            [143, 255, 140],
            [204, 255, 4],
            [255, 51, 7],
            [204, 70, 3],
            [0, 102, 200],
            [61, 230, 250],
            [255, 6, 51],
            [11, 102, 255],
            [255, 7, 71],
            [255, 9, 224],
            [9, 7, 230],
            [220, 220, 220],
            [255, 9, 92],
            [112, 9, 255],
            [8, 255, 214],
            [7, 255, 224],
            [255, 184, 6],
            [10, 255, 71],
            [255, 41, 10],
            [7, 255, 255],
            [224, 255, 8],
            [102, 8, 255],
            [255, 61, 6],
            [255, 194, 7],
            [255, 122, 8],
            [0, 255, 20],
            [255, 8, 41],
            [255, 5, 153],
            [6, 51, 255],
            [235, 12, 255],
            [160, 150, 20],
            [0, 163, 255],
            [140, 140, 140],
            [250, 10, 15],
            [20, 255, 0],
            [31, 255, 0],
            [255, 31, 0],
            [255, 224, 0],
            [153, 255, 0],
            [0, 0, 255],
            [255, 71, 0],
            [0, 235, 255],
            [0, 173, 255],
            [31, 0, 255],
            [11, 200, 200],
            [255, 82, 0],
            [0, 255, 245],
            [0, 61, 255],
            [0, 255, 112],
            [0, 255, 133],
            [255, 0, 0],
            [255, 163, 0],
            [255, 102, 0],
            [194, 255, 0],
            [0, 143, 255],
            [51, 255, 0],
            [0, 82, 255],
            [0, 255, 41],
            [0, 255, 173],
            [10, 0, 255],
            [173, 255, 0],
            [0, 255, 153],
            [255, 92, 0],
            [255, 0, 255],
            [255, 0, 245],
            [255, 0, 102],
            [255, 173, 0],
            [255, 0, 20],
            [255, 184, 184],
            [0, 31, 255],
            [0, 255, 61],
            [0, 71, 255],
            [255, 0, 204],
            [0, 255, 194],
            [0, 255, 82],
            [0, 10, 255],
            [0, 112, 255],
            [51, 0, 255],
            [0, 194, 255],
            [0, 122, 255],
            [0, 255, 163],
            [255, 153, 0],
            [0, 255, 10],
            [255, 112, 0],
            [143, 255, 0],
            [82, 0, 255],
            [163, 255, 0],
            [255, 235, 0],
            [8, 184, 170],
            [133, 0, 255],
            [0, 255, 92],
            [184, 0, 255],
            [255, 0, 31],
            [0, 184, 255],
            [0, 214, 255],
            [255, 0, 112],
            [92, 255, 0],
            [0, 224, 255],
            [112, 224, 255],
            [70, 184, 160],
            [163, 0, 255],
            [153, 0, 255],
            [71, 255, 0],
            [255, 0, 163],
            [255, 204, 0],
            [255, 0, 143],
            [0, 255, 235],
            [133, 255, 0],
            [255, 0, 235],
            [245, 0, 255],
            [255, 0, 122],
            [255, 245, 0],
            [10, 190, 212],
            [214, 255, 0],
            [0, 204, 255],
            [20, 0, 255],
            [255, 255, 0],
            [0, 153, 255],
            [0, 41, 255],
            [0, 255, 204],
            [41, 0, 255],
            [41, 255, 0],
            [173, 0, 255],
            [0, 245, 255],
            [71, 0, 255],
            [122, 0, 255],
            [0, 255, 184],
            [0, 92, 255],
            [184, 255, 0],
            [0, 133, 255],
            [255, 214, 0],
            [25, 194, 194],
            [102, 255, 0],
            [92, 0, 255],
        ]

    @prompts(
        name="Segmentation On Image",
        description="useful when you want to detect segmentations of the image. "
        "like: segment this image, or generate segmentations on this image, "
        "or peform segmentation on this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        # height, width, 3
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(self.ade_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        segmentation = Image.fromarray(color_seg)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        segmentation.save(updated_image_path)
        print(
            f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}"
        )
        return updated_image_path


class SegText2Image:
    def __init__(self, device):
        print("Initializing SegText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-seg", torch_dtype=self.torch_dtype
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Segmentations",
        description="useful when you want to generate a new real image from both the user desciption and segmentations. "
        "like: generate a real image of a object or something from this segmentation image, "
        "or generate a new real image of a object or something from these segmentations. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Image2Depth:
    def __init__(self, device):
        print("Initializing Image2Depth")
        self.depth_estimator = pipeline("depth-estimation")

    @prompts(
        name="Predict Depth On Image",
        description="useful when you want to detect depth of the image. like: generate the depth from this image, "
        "or detect the depth map on this image, or predict the depth for this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        depth = self.depth_estimator(image)["depth"]
        depth = np.array(depth)
        depth = depth[:, :, None]
        depth = np.concatenate([depth, depth, depth], axis=2)
        depth = Image.fromarray(depth)
        updated_image_path = get_new_image_name(inputs, func_name="depth")
        depth.save(updated_image_path)
        print(
            f"\nProcessed Image2Depth, Input Image: {inputs}, Output Depth: {updated_image_path}"
        )
        return updated_image_path


class DepthText2Image:
    def __init__(self, device):
        print("Initializing DepthText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-depth",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Depth",
        description="useful when you want to generate a new real image from both the user desciption and depth image. "
        "like: generate a real image of a object or something from this depth image, "
        "or generate a new real image of a object or something from the depth map. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="depth2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed DepthText2Image, Input Depth: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class Image2Normal:
    def __init__(self, device):
        print("Initializing Image2Normal")
        self.depth_estimator = pipeline(
            "depth-estimation", model="Intel/dpt-hybrid-midas"
        )
        self.bg_threhold = 0.4

    @prompts(
        name="Predict Normal Map On Image",
        description="useful when you want to detect norm map of the image. "
        "like: generate normal map from this image, or predict normal map of this image. "
        "The input to this tool should be a string, representing the image_path",
    )
    def inference(self, inputs):
        image = Image.open(inputs)
        original_size = image.size
        image = self.depth_estimator(image)["predicted_depth"][0]
        image = image.numpy()
        image_depth = image.copy()
        image_depth -= np.min(image_depth)
        image_depth /= np.max(image_depth)
        x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        x[image_depth < self.bg_threhold] = 0
        y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        y[image_depth < self.bg_threhold] = 0
        z = np.ones_like(x) * np.pi * 2.0
        image = np.stack([x, y, z], axis=2)
        image /= np.sum(image**2.0, axis=2, keepdims=True) ** 0.5
        image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image = image.resize(original_size)
        updated_image_path = get_new_image_name(inputs, func_name="normal-map")
        image.save(updated_image_path)
        print(
            f"\nProcessed Image2Normal, Input Image: {inputs}, Output Depth: {updated_image_path}"
        )
        return updated_image_path


class NormalText2Image:
    def __init__(self, device):
        print("Initializing NormalText2Image to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-normal",
            torch_dtype=self.torch_dtype,
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            safety_checker=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = "best quality, extremely detailed"
        self.n_prompt = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,"
            " fewer digits, cropped, worst quality, low quality"
        )

    @prompts(
        name="Generate Image Condition On Normal Map",
        description="useful when you want to generate a new real image from both the user desciption and normal map. "
        "like: generate a real image of a object or something from this normal map, "
        "or generate a new real image of a object or something from the normal map. "
        "The input to this tool should be a comma seperated string of two, "
        "representing the image_path and the user description",
    )
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ",".join(
            inputs.split(",")[1:]
        )
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ", " + self.a_prompt
        image = self.pipe(
            prompt,
            image,
            num_inference_steps=20,
            eta=0.0,
            negative_prompt=self.n_prompt,
            guidance_scale=9.0,
        ).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="normal2image")
        image.save(updated_image_path)
        print(
            f"\nProcessed NormalText2Image, Input Normal: {image_path}, Input Text: {instruct_text}, "
            f"Output Image: {updated_image_path}"
        )
        return updated_image_path


class VisualQuestionAnswering:
    def __init__(self, device):
        print("Initializing VisualQuestionAnswering to %s" % device)
        self.torch_dtype = torch.float16 if "cuda" in device else torch.float32
        self.device = device
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base", torch_dtype=self.torch_dtype
        ).to(self.device)

    @prompts(
        name="Answer Question About The Image",
        description="useful when you need an answer for a question based on an image. "
        "like: what is the background color of the last image, how many cats in this figure, what is in this figure. "
        "The input to this tool should be a comma seperated string of two, representing the image_path and the question",
    )
    def inference(self, inputs):
        image_path, question = inputs.split(",")
        raw_image = Image.open(image_path).convert("RGB")
        inputs = self.processor(raw_image, question, return_tensors="pt").to(
            self.device, self.torch_dtype
        )
        out = self.model.generate(**inputs)
        answer = self.processor.decode(out[0], skip_special_tokens=True)
        print(
            f"\nProcessed VisualQuestionAnswering, Input Image: {image_path}, Input Question: {question}, "
            f"Output Answer: {answer}"
        )
        return answer
