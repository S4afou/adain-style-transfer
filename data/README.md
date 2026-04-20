# Datasets

Both datasets were mounted directly from Kaggle, so no download step appears in the report.
Neither is redistributed here.

## MS-COCO 2017 (content images)

- Source: https://cocodataset.org
- Contents: about 118,000 training images of natural scenes and objects
- Access: free, Creative Commons attribution for the images
- Used as: the content image for each training pair

## WikiArt (style images)

- Source: Kaggle, "WikiArt dataset" (painting collections grouped by artist and movement)
- Contents: paintings spanning a wide range of movements and painters
- Access: check the licence of the specific Kaggle mirror before reuse
- Used as: the style image for each training pair

## Sampling

Each training step draws one content image and one style image independently, so the model
never learns a fixed style. That is the property that makes the decoder style-agnostic, and
it is why a single trained decoder can accept any style image at inference time.

## Checkpoints

Decoder weights, optimizer state and the loss history were pushed to a Hugging Face model
repository at each checkpoint interval so that training could resume across ephemeral
Kaggle sessions. No weights are committed to this repository.
