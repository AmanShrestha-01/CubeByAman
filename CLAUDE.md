# Cube Coach
A full-stack web app: the camera reads a physical Rubik's cube, an ML
model classifies the sticker colors, a solver returns the moves, and the
app coaches the user through solving it.

## About me
I know Next.js, TypeScript, Supabase. I finished one classical ML project
(chess winner predictor, scikit-learn). This is my FIRST deep learning project.

## Phases (do NOT jump ahead)
1. ml/  — sticker colour classifier (CNN) + a non-ML baseline  ← WE ARE HERE
2. ml/  — OpenCV: find the cube face, correct perspective, split into 3x3
3. ml/  — capture 6 faces, validate state, solve with kociemba
4. api/ — FastAPI service wrapping the model and solver
5. web/ — Next.js + Supabase app with camera scanning
6.     — polish, 3D preview, deploy

## Phase 1 plan (sticker colour classifier)
0. Environment check ✅
1. Tensors: how an image becomes numbers ✅
2. Dataset: collect and label sticker images  ← CURRENT STEP. Keep a held-out test set that
   includes BAD LIGHTING (dim room, warm/yellow bulb, shadow, glare).
3. **Non-ML baseline (REQUIRED before any CNN):** classify with HSV thresholds
   in OpenCV. Measure accuracy on the test set, both overall and on the
   bad-lighting subset, and record the numbers here.
4. CNN: layers and the forward pass.
5. Training loop: loss, gradients, optimizer.
6. Evaluation: CNN vs baseline on the SAME test sets (accuracy + confusion
   matrix), especially under bad lighting. The CNN must beat the baseline to
   justify its complexity.

Rule: do not start step 4 until the baseline numbers are recorded below.

### Results log
| Model | Overall acc. | Bad-lighting acc. | Notes |
|-------|--------------|-------------------|-------|
| HSV baseline | – | – | not yet measured |
| CNN | – | – | not yet trained |

## Teaching rules (most important)
- I'm here to LEARN. Explain the concept in plain words BEFORE any code.
- Teach the concept like you would to a 15 year old with no prior knowledge of machine learning. Use simple language and real-world analogies.
- One concept at a time. Wait for me to run each cell before continuing.
- Let me type the key lines myself (training loop, forward pass).
- Define every new deep learning term in one sentence when it first appears.
- After each step, ask me one question to check I understood.
- Never skip ahead to later phases.
- This is more learning than building. With every prompt, give me a detailed
  guide to what is going on: what you are doing, why, and how it fits the
  bigger picture.
- Every commit message must teach, not just list changes: a short summary line,
  then a body explaining WHAT changed, WHY, and the concepts behind it, so
  `git log` reads like a study journal.

## Phase 1 stack
Python 3.11, PyTorch, torchvision, OpenCV, matplotlib, Jupyter, .venv

## Setup
```bash
cd ml
source .venv/bin/activate          # activate the virtual environment
jupyter lab color_classifier.ipynb # or open the notebook in VS Code
```
