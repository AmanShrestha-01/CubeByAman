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
2. Dataset: collect and label sticker images ✅. Keep a held-out test set that
   includes BAD LIGHTING (dim room, warm/yellow bulb, shadow, glare).
3. ← CURRENT STEP. **Non-ML baseline (REQUIRED before any CNN):** classify with HSV thresholds
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
| HSV baseline | 100% (318/318 test_normal) | 100% (212/212, dim only) | train 99.5%: 2 glare blues → white. Rules: S<71 white; H<8 red, <19 orange, <37 yellow, <79 green, <143 blue, else red. Fences = midpoints of train ranges, outliers ignored. test_hard_light 100% (159/159: glare, warm, shadow 53/53 each). BUT margins are thin: dim oranges sat at H=8, exactly ON the red/orange fence (0 steps to spare); glare/warm/shadow closest calls were 3 steps, all red/orange. Accuracy is saturated, so compare models by margin/stress tests, not accuracy alone |
| CNN | – | – | not yet trained |

### Dataset (photos are gitignored, so this is the record)
Stickerless cube (no black borders); the white centre has a logo. Solved
cube, so one photo = one face = 9 stickers of the folder's colour.
`ml/data/raw/photos/<split>/<colour>/IMG_*.JPG`, 3024x4032 iPhone JPGs;
the face fills only part of the frame, so it must be boxed before splitting.

| Split | Per colour | Total | Conditions |
|-------|-----------|-------|------------|
| train | 8 | 48 | desk room, varied positions |
| test_normal | 6 | 36 | different room; intended bad-light shots came out normal (phone auto-exposure/white balance) |
| test_bad_light | 4 | 24 | third spot, genuinely dim (~20-40% darker); NO shadow, glare or warm tint yet |
| test_hard_light | 3 | 18 | one photo per colour per condition. Glare (wall): 7955-7960. Warm lamp (basket): 7953, 7961-7965. Shadow (carpet): 7966-7971 |

Stickers: `ml/data/stickers/<split>/<colour>/IMG_xxxx_r<row>c<col>.png`, 64x64,
954 total (train 424, test_normal 318, test_bad_light 212). Rebuilt any time by
`tools/make_stickers.py` from the hand-drawn boxes in `ml/data/boxes.json`.
White centres (logo) are skipped. Known hard example: train/blue/IMG_7873_r1c0
is glare that looks white but is correctly labelled blue.

## Teaching rules (most important)
- I'm here to LEARN. Explain the concept in plain words BEFORE any code.
- Teach the concept like you would to a 15 year old with no prior knowledge of machine learning. Use simple language and real-world analogies.
- English is not my first language. Use short sentences and simple, common
  words. Show a small example with real numbers instead of long explanations.
  Keep each message short: one idea, then stop.
- One concept at a time. Wait for me to run each cell before continuing.
- Let me type the key lines myself (training loop, forward pass).
- Define every new deep learning term in one sentence when it first appears.
- After each step, ask me one question to check I understood.
- Whenever I ask for an explanation or recap, write it so I can retell it to
  three audiences: non-technical people, coding friends, and software
  recruiters. Cover the procedure step by step, the tools used and why, and
  how the image/colour recognition actually works. Be honest about what's
  built vs. not yet (e.g. rules vs. a trained model, manual vs. automatic).
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
