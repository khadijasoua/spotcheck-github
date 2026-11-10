# SpotCheck: learning notes

Questions we asked while building SpotCheck, with the answers, grouped by file. Written as study notes for the course; the code itself is documented in each file.

**Contents:** [Shared code](#shared-code-commonpy) · [The data](#the-data-explorepy) · [Training](#training-trainpy) · [Evaluation](#evaluation-evaluatepy) · [The referral threshold](#the-referral-threshold-thresholdpy) · [Grad-CAM](#grad-cam-heatmappy)

## Shared code (`common.py`)

**What is common.py, and why do we need it?**

It is the shared toolbox. The other files need the same things: loading data, building the model, getting predictions. Writing them once here means every file uses exactly the same settings, and a change (e.g. the image size) happens in one place only.

**Why are the photos "normalised" with those six numbers?**

ResNet-18 was pretrained on ImageNet photos prepared this way: each colour shifted by its ImageNet average and divided by its standard deviation. Our photos must be on the same scale, or the pretrained filters receive numbers they never learned to handle.

**What does "transfer learning" mean in this file?**

build_model() takes ResNet-18, already trained on 1.2 million everyday photos, keeps everything that detects edges, textures and shapes, and only replaces the last layer (1,000 everyday classes -> our 7 skin types). Training then fine-tunes it on skin photos. Without this, 7,007 photos would be far too few.

**What does softmax do?**

The model's last layer gives 7 raw scores (any numbers, even negative). Softmax turns them into probabilities between 0 and 1 that add up to 1, keeping the order: the highest score gets the highest probability.

**Why is there no random flipping for validation and test photos?**

Flipping only adds variety while LEARNING. When we measure the model, every photo must be shown exactly as it is, so the results are fair and repeatable.

## The data (`explore.py`)

**I thought the dataset had about 10,000 photos. Why only 7,007?**

explore.py only loads the TRAINING part (split="train"). The 10,015 photos are pre-split by the dataset's creators: 7,007 training, 1,003 validation, 2,005 test.

**How does it know how many photos to download?**

It downloads one file (dermamnist_128.npz) containing all three parts. split="train" then opens only the training part. Later files open the validation or test part of the same file, without downloading again.

**Who decided the split?**

The MedMNIST creators. Because everyone uses the same test photos, our results can be fairly compared with published results.

**The download failed with CERTIFICATE_VERIFY_FAILED. Why?**

Python installed from python.org on a Mac doesn't trust website certificates yet. Running "Install Certificates.command" once (in Applications -> Python 3.x) fixes it.

## Training (`train.py`)

**What is a CNN?**

A convolutional neural network: the standard model type for images. It slides small grids of learned weights (filters) over the image. Each filter gives a high value wherever its pattern appears. Early layers detect edges and colours, deeper layers combine them into textures and shapes, and the last layer turns everything into 7 scores, one per skin type.

**How does it train?**

A loop, repeated for every batch of 64 photos:
1. Guess: model(x) gives 7 scores per photo.
2. Measure the mistake: the loss (cross-entropy) is big when the model gives a low probability to the correct type.
3. Find the blame: loss.backward() computes, for every weight, how changing it would change the loss (the gradient).
4. Nudge: optimizer.step() moves every weight a tiny step in the direction that lowers the loss (learning rate 0.0001).

**What is an epoch?**

One full pass through all 7,007 training photos. They go in batches of 64, so one epoch = 110 batches = 110 small weight updates. 5 epochs = 550 updates in total.

**Do I run train.py 5 times?**

No. I run it once. The line "for epoch in range(EPOCHS)" repeats the training 5 times automatically. It is run a second time only with the other WEIGHTED setting, to train the second model.

**Why not train for many more epochs?**

Too many epochs and the model memorises the training photos instead of learning general patterns (overfitting). It gets better on training photos but worse on new ones. That's why the code checks the validation photos after every epoch and saves only the best version.

**If the loss keeps going down, why did validation accuracy drop at epoch 3?**

The loss is measured on the TRAINING photos, the accuracy on the VALIDATION photos the model never learns from. Getting better on the training photos doesn't automatically mean getting better on new ones. Three reasons for the dip:
1. Early overfitting: the model starts learning details specific to the training photos (like memorising exam answers).
2. Chance: the validation set has few photos of the rare types, so a few extra mistakes move balanced accuracy a lot.
3. Loss also rewards being MORE CONFIDENT about answers it already gets right; accuracy only counts right or wrong. Lesson: never judge a model by its training loss. Judge it on photos it hasn't learned from.

**Does it save the last epoch or the best one?**

The best one: whenever the validation balanced accuracy beats the best so far, the file is overwritten. The plain model kept epoch 4, the weighted model epoch 5. The two runs save to different files, so the second run never overwrites the first model.

**"Weights" means two things here. What's the difference?**

1. The model's weights: the ~11 million numbers inside the network that training changes. Saved in the .pt files.
2. Class weights: how much a mistake on each TYPE counts in the loss (WEIGHTED = True). An extra setting to fight the imbalance.

**Does the weighted run take longer?**

No. Same photos, same batches, same epochs; multiplying the loss by a weight costs practically nothing.

## Evaluation (`evaluate.py`)

**What is a confusion matrix?**

A table that counts the model's answers. Each ROW is what a photo really is, each COLUMN is what the model said. The diagonal holds the correct answers; every other cell is a specific mix-up, e.g. "47 melanomas called mole".

**What is the difference between recall and precision?**

Read them in the melanoma row and column of the confusion matrix:
- Recall (melanoma ROW): of all real melanomas, how many were caught? 141 / 223 = 63%. Missing a cancer is the worst mistake.
- Precision (melanoma COLUMN): of all melanoma alarms, how many were real? 141 / 243 = 58%. The rest are false alarms.

**Why is 83% accuracy not good enough on its own?**

Always guessing "mole" already gives 67%. And accuracy hides which mistakes are made: the 83% model still missed more than 1 in 3 melanomas. For a cancer tool, melanoma recall matters most.

**Why run the same file twice?**

The model name after the command decides which model is loaded. Testing both on the same test photos makes the comparison fair, and each run saves its own confusion-matrix picture.

**Which accuracy is "my model's" accuracy?**

The TEST numbers here: 83.0% (plain) and 77.3% (weighted). The numbers printed during training are on the validation photos, which were used to choose the best epoch, so they are not final.

**Why did I get "KeyboardInterrupt" at the end?**

plt.show() waits until the picture window is closed. Starting another program while it waits interrupts it. Harmless: the picture was already saved. Close the window to finish cleanly.

## The referral threshold (`threshold.py`)

**What does the threshold change?**

Only WHEN the alarm goes off. The model and its probabilities stay exactly the same; no retraining. Example: mole 55%, melanoma 35%. "Highest wins" says mole, no alarm. With a threshold of 0.2 the alarm goes off, because 35% >= 20%.

**Why choose the threshold on the validation set, not the test set?**

Choosing it by looking at the test results would be tuning to the exam answers, so the final number would look better than it really is. The test set is used once, after the choice is fixed.

**Is there a mathematically correct threshold?**

No. It is a judgement about which mistake is worse: a missed cancer or an unnecessary referral. That is why the decision belongs to people (here: us, reasoning as a GP), not the model.

## Grad-CAM (`heatmap.py`)

**Does the heatmap decide whether a melanoma is caught or missed?**

No. The model gives the probabilities and the threshold decides the alarm. Grad-CAM comes afterwards: like asking a doctor "show me where you looked". It changes nothing about the result, but it helps us tell "looked at the wrong place" apart from "looked at the right place but judged it wrongly".

**How does Grad-CAM work, roughly?**

It takes the 512 feature maps of the last convolutional block and asks, for each map, how much it pushed the score of the predicted type up (using the gradient). It adds the maps together, weighted by that importance, keeps only the positive part, and stretches the 4 x 4 result over the photo as colours.

**Why look at missed melanomas at all?**

Mistakes teach more than successes. They show where the model is weak (faint, small, atypical lesions), which is exactly what the evaluation and the "next steps" are about.

**Why do all the heatmaps look so blurry?**

The model's last layer sees the photo as a 4 x 4 grid (128 pixels halved 5 times). Grad-CAM can't be sharper than that grid.

## The seven lesion types

| Class | Malignant / premalignant | Description |
|---|---|---|
| Actinic keratoses and intraepithelial carcinoma | ✓ | Rough, scaly sun-damage lesions; some are an early skin cancer confined to the top layer of the skin. |
| Basal cell carcinoma | ✓ | The most common skin cancer; grows slowly and rarely spreads, but damages surrounding tissue if untreated. |
| Benign keratosis-like lesions | | Harmless growths such as age spots and wart-like lesions; can resemble melanoma. |
| Dermatofibroma | | A small, firm, harmless nodule, often on the legs. |
| Melanoma | ✓ | Cancer of the pigment cells; the most dangerous type because it can spread. |
| Melanocytic nevi | | Ordinary moles; 67% of the training data. |
| Vascular lesions | | Harmless lesions made of small blood vessels, e.g. cherry angiomas. |
