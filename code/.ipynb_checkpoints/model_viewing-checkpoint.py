import sys

import data
import data.snli
import settings
model, dataset = data.snli.load_for_analysis(
    settings.MODEL,
    settings.DATA,
    model_type=settings.MODEL_TYPE,
    cuda=settings.CUDA,
)

print(model)