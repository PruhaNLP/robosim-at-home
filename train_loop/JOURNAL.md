# Журнал итераций кастомного цикла обучения

## Итерация 0 — бейзлайн (lerobot.scripts.lerobot_train)

Цель: зафиксировать скорость и поведение официального цикла на V100 / SmolVLA / локальном датасете `data/lerobot/smolvla` (9 эпизодов, front+wrist, 512², 2775 кадров).

Условия (как в UI на этой машине):
- `policy=lerobot/smolvla_base`
- `train_expert_only=true`, `freeze_vision_encoder=true`
- AMP выключен (V100, CC 7.0, у lerobot AMP = BF16)
- `empty_cameras=1` (модель всегда видит 3 камеры)
- rename: front→camera1, wrist→camera2
- batch=4, steps=8, log_freq=1, без чекпоинтов

Метрики: step time, data_s, updt_s, smp/s, mem_gb, loss.

Результат (`results/baseline.json`):
- wall: 38.5 с на 8 шагов (включая загрузку модели)
- после прогрева: **updt_s=0.558**, data_s=0.003, **smp/s=7.0**, mem=2.43 ГБ
- первый шаг: data_s=7.54 с (старт DataLoader workers)
- loss на шаге 8: 0.831
- узкое место — update (forward+backward без AMP, 3 камеры включая empty)

## Итерация 1 — свой цикл

Цели:
- тот же CLI, что у `lerobot.scripts.lerobot_train`
- FP16 AMP на V100
- без Accelerate на одном GPU
- пропуск отсутствующих камер + `episode_cameras.json`
- группировка батчей по набору камер
- GPU uint8→float
- `zero_grad(set_to_none=True)`

Результат (`results/ours.json`, те же 8 шагов / batch 4 / empty_cameras=1):
- wall: 38.3 с
- после прогрева: **updt_s=0.316** (−43%), **smp/s=12.6** (×1.8), data_s=0.003, mem=3.96 ГБ
- FP16 AMP на V100 работает (мастер-веса FP32 + GradScaler)
- цикл живой, тот же CLI

## Итерация 2 — меньше камер + variable cameras

- `empty_cameras=0` (не гонять dummy-view через VLM)
- проверка батчей с разным числом камер

`empty_cameras=0`: **updt_s=0.280** (−50% к бейзлайну), **smp/s=14.3** (×2.0), mem=3.70 ГБ.
Variable-camera forward: mixed batch схлопывается в пересечение камер, same-set сохраняет все, loss считается.
Mixed-camera train (разные наборы камер по эпизодам): **ok**, updt_s=0.268, smp/s=15.0.

## Итерация 3 — кэш эмбеддингов замороженного vision

При `freeze_vision_encoder` кэшировать SigLIP-эмбеддинги по (episode, frame, camera). Выигрыш после первого прохода по датасету.

800 шагов, batch 4:
- до прогрева кэша (шаг 680): updt_s=0.288, data_s≈0.05–0.13, ~2.7 step/s, smp/s≈11
- после 1 эпохи (5550 ключей, шаг 720+): **updt_s=0.245**, data_s=0.003, **~4.0 step/s**, **smp/s=16**
- loss 0.54 → 0.21
- vs бейзлайн 7 smp/s: **×2.3** на прогретом цикле

## Сводка vs lerobot

| режим | updt_s | smp/s | mem_gb |
|---|---|---|---|
| бейзлайн AMP off, empty=1 | 0.558 | 7.0 | 2.43 |
| наш FP16, empty=1 | 0.316 | 12.6 | 3.96 |
| наш FP16, empty=0 | 0.280 | 14.3 | 3.70 |
| наш + кэши после 1 эпохи | 0.245 | 16.0 | 4.40 |
| бейзлайн batch 8 | 0.913 | 8.6 | 3.59 |
| наш FP16 empty=0 batch 8 | 0.344 | 18.4 | 4.44 |
| наш FP16 empty=0 batch 16 | 0.413 | 25.0 | 6.07 |

