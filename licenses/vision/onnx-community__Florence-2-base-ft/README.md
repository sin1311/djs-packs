---
base_model: microsoft/Florence-2-base-ft
library_name: transformers.js
license: mit
pipeline_tag: image-text-to-text
tags:
- vision
- text-generation
- text2text-generation
- image-to-text
---

https://huggingface.co/microsoft/Florence-2-base-ft with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Perform image captioning with `onnx-community/Florence-2-base-ft`.
```js
import {
    Florence2ForConditionalGeneration,
    AutoProcessor,
    load_image,
} from '@huggingface/transformers';

// Load model, processor, and tokenizer
const model_id = 'onnx-community/Florence-2-base-ft';
const model = await Florence2ForConditionalGeneration.from_pretrained(model_id, { dtype: 'fp32' });
const processor = await AutoProcessor.from_pretrained(model_id);

// Load image and prepare vision inputs
const url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg';
const image = await load_image(url);

// Specify task and prepare text inputs
const task = '<MORE_DETAILED_CAPTION>';
const prompts = processor.construct_prompts(task);

// Pre-process the image and text inputs
const inputs = await processor(image, prompts);

// Generate text
const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: 100,
});

// Decode generated text
const generated_text = processor.batch_decode(generated_ids, { skip_special_tokens: false })[0];

// Post-process the generated text
const result = processor.post_process_generation(generated_text, task, image.size);
console.log(result);
// { '<MORE_DETAILED_CAPTION>': 'A green car is parked in front of a tan building. There is a brown door on the building behind the car. There are two windows on the front of the building. ' }
```

We also released an online demo, which you can try yourself: https://huggingface.co/spaces/Xenova/florence2-webgpu


<video controls autoplay src="https://cdn-uploads.huggingface.co/production/uploads/61b253b7ac5ecaae3d1efe0c/BJj3jQXNqS_7Nt2MSb2ss.mp4"></video>

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).

