from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_min_step(log_dir):
    ea = EventAccumulator(log_dir)
    ea.Reload()
    tag = "loss"
    events = ea.Scalars(tag)
    min_event = min(events, key=lambda e: e.value)
    return min_event.step



# ログファイルがあるディレクトリ（runs/...）
log_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\Robomech_Diffusion\all_use\logs\loss_test" # ← TensorBoardのlogdirと同じ

# EventAccumulator を初期化してロード
ea = EventAccumulator(log_dir)
ea.Reload()

# 目的のスカラー名（例： 'loss/test' ）
tag = "loss"

# 全ステップ分のスカラーを取得（step, wall_time, value）
events = ea.Scalars(tag)

# 最小値とそのステップを計算
min_event = min(events, key=lambda e: e.value)

print(f"最小 loss = {min_event.value:.6f} @ step = {min_event.step}")
