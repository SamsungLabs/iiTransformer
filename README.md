# iiTransformer
This is an official repository for **"iiTransformer: A Unified Approach to Exploiting Local and Non-Local Information for Image Restoration"** (Kang et al., BMVC 2022) **(Spotlight)**

Code will be released soon! 

## Introduction
<picture> <source media="" srcset="/assets/images/intro.pdf"> </picture>
iiTransformer exploits local and non-local information within an image by treating pixels and patches as tokens, resp., in a Transformer layer. 

<picture> <source media="" srcset="/assets/images/aMSA_eMSA.pdf"> </picture>
The switch between intra MSA (aMSA) and inter MSA (eMSA) is determined by the use of a reshape operation within the MSA module of a Transformer layer.
