---
base_model: google/siglip-base-patch16-224
library_name: transformers.js
pipeline_tag: zero-shot-image-classification
---

https://huggingface.co/google/siglip-base-patch16-224 with ONNX weights to be compatible with Transformers.js.

## Usage (Transformers.js)

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:
```bash
npm i @huggingface/transformers
```

**Example:** Zero-shot image classification w/ `Xenova/siglip-base-patch16-224`:
```js
import { pipeline } from '@huggingface/transformers';

const classifier = await pipeline('zero-shot-image-classification', 'Xenova/siglip-base-patch16-224');
const url = 'http://images.cocodataset.org/val2017/000000039769.jpg';
const output = await classifier(url, ['2 cats', '2 dogs'], {
    hypothesis_template: 'a photo of {}',
});
console.log(output);
// [
//   { score: 0.16770583391189575, label: '2 cats' },
//   { score: 0.000022096000975579955, label: '2 dogs' }
// ]
```

**Example:** Compute text embeddings with `SiglipTextModel`.

```javascript
import { AutoTokenizer, SiglipTextModel } from '@huggingface/transformers';

// Load tokenizer and text model
const tokenizer = await AutoTokenizer.from_pretrained('Xenova/siglip-base-patch16-224');
const text_model = await SiglipTextModel.from_pretrained('Xenova/siglip-base-patch16-224');

// Run tokenization
const texts = ['a photo of 2 cats', 'a photo of 2 dogs'];
const text_inputs = tokenizer(texts, { padding: 'max_length', truncation: true });

// Compute embeddings
const { pooler_output } = await text_model(text_inputs);
// Tensor {
//   dims: [ 2, 768 ],
//   type: 'float32',
//   data: Float32Array(1536) [ ... ],
//   size: 1536
// }
```

**Example:** Compute vision embeddings with `SiglipVisionModel`.

```javascript
import { AutoProcessor, SiglipVisionModel, RawImage} from '@huggingface/transformers';

// Load processor and vision model
const processor = await AutoProcessor.from_pretrained('Xenova/siglip-base-patch16-224');
const vision_model = await SiglipVisionModel.from_pretrained('Xenova/siglip-base-patch16-224');

// Read image and run processor
const image = await RawImage.read('https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg');
const image_inputs = await processor(image);

// Compute embeddings
const { pooler_output } = await vision_model(image_inputs);
// Tensor {
//   dims: [ 1, 768 ],
//   type: 'float32',
//   data: Float32Array(768) [ ... ],
//   size: 768
// }
```

---

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).