import torch
import torch.nn as nn
from myclass import myfunction, Mydataset
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =========================
# 1) 教師(どっちが近いか)を作る
# =========================
@torch.no_grad()
def teacher_label_std(mu_std, y_std):
    # mu_std: (N,2,3) 標準化空間, y_std: (N,3)
    d1 = torch.norm(mu_std[:,0,:] - y_std, dim=1)
    d2 = torch.norm(mu_std[:,1,:] - y_std, dim=1)
    label = (d2 < d1).long()        # 0: mu1が近い, 1: mu2が近い
    margin = (d1 - d2).abs()        # どれくらい差がついたか
    return label, margin, d1, d2

# =========================
# 2) 診断（学習時と同じ特徴で）
# =========================
@torch.no_grad()
def quick_diagnose(mdnn, selector, t3_std, m9_std, y_std):
    # mdnn: TorchScript MDN, selector: TorchScript SelectorNet
    # t3_std: (N,3) 標準化済みの回転3ch
    # m9_std: (N,9) 標準化済みの磁気9ch
    # y_std : (N,3) 標準化済みの正解座標
    pi, mu_std, sigma = mdnn(t3_std)                  # mu_std:(N,2,3)

    # ====== 学習時と同じ特徴量の作り方 ======
    feats = torch.cat([mu_std[:,0,:], mu_std[:,1,:], m9_std], dim=1)  # (N,15)

    logits, _ = selector(feats)
    pick = logits.argmax(dim=1)                      # (N,)

    teacher, margin, d1, d2 = teacher_label_std(mu_std, y_std)

    # ① 教師ラベルの偏り
    p1 = (teacher==1).float().mean().item()
    # ② 教師に対する精度（= 本来のゴール）
    acc = (pick==teacher).float().mean().item()
    # ③ 自信の高い半分だけ（距離差marginの上位50%）での精度
    thr = margin.median()
    mask = margin > thr
    acc_conf = (pick[mask]==teacher[mask]).float().mean().item() if mask.any() else float('nan')
    # ④ 出力崩壊チェック（常に同じクラスを言ってないか）
    frac_pick1 = (pick==1).float().mean().item()
    # ⑤ ベースライン：常に多数派を言った時の精度（偏りの指標）
    baseline = max(p1, 1.0-p1)

    print("=== Quick Diagnose (selector vs teacher) ===")
    print(f"teacher P(label=1)            = {p1:.3f}")
    print(f"baseline(always-majority)     = {baseline:.3f}")
    print(f"selector acc vs teacher (all) = {acc:.3f}")
    print(f"selector acc (confident half) = {acc_conf:.3f}")
    print(f"selector picks class-1 ratio  = {frac_pick1:.3f}")
    # 参考に数件だけ詳細を出す
    for i in range(min(10, t3_std.shape[0])):
        print(f"[{i:04d}] pick={int(pick[i].item())} teacher={int(teacher[i].item())} "
              f"d1={d1[i].item():.3f} d2={d2[i].item():.3f} margin={margin[i].item():.3f}")

# =========================
# 3) 前処理（学習時と完全一致）
# =========================
def load_std_data_for_diagnosis(data_path, selector_result_dir):
    # Selector 学習時に保存した scaler を使う（重要！）
    scaler_path = myfunction.find_pickle_files("scaler", selector_result_dir)
    scaler = myfunction.load_pickle(scaler_path)

    x_mean = torch.tensor(scaler['x_mean'])
    x_std  = torch.tensor(scaler['x_std'])
    y_mean = torch.tensor(scaler['y_mean'])
    y_std  = torch.tensor(scaler['y_std'])
    fitA   = scaler['fitA']

    # 生データ読み込み（Selector 学習時と同じモードで）
    x_data, y_data, typedf = myfunction.read_pickle_to_torch(
        data_path, motor_angle=True, motor_force=True, magsensor=True
    )
    y_last3 = y_data[:, -3:]

    # 学習時と同じ整列→標準化
    alphaA = torch.ones_like(fitA.amp)
    xA_proc   = Mydataset.apply_align_torch(x_data, fitA, alphaA)
    std_x_all = Mydataset.apply_standardize_torch(xA_proc, x_mean, x_std)
    std_y_all = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)

    # NaN除去も学習時と同様に
    mask = torch.isfinite(x_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
    std_x = std_x_all[mask]
    std_y = std_y_all[mask]

    # 回転・力・磁気に分割（学習時と同じ 3,3,9）
    t3_std, f3_std, m9_std = torch.split(std_x, [3,3,9], dim=1)

    # デバイスへ
    return (t3_std.to(device).float(),
            m9_std.to(device).float(),
            std_y.to(device).float(),
            y_mean.to(device).float(),
            y_std.to(device).float())

# =========================
# 4) 実行ブロック（パスだけ直してください）
# =========================
if __name__ == "__main__":
    # ---- パス設定 ----
    MDNpath      = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
    selectorpath = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\select2\selector.pth"
    selector_dir = os.path.dirname(selectorpath)

    # 診断したいデータ（学習に使ったもの or 評価したいファイル）
    datapath     = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
    # 例：テスト側を見るなら ↓ を使う
    # datapath  = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit10kaifortestnew.pickle"

    # ---- モデル読み込み（TorchScript）----
    mdn = torch.jit.load(MDNpath, map_location=device).eval()
    selector = torch.jit.load(selectorpath, map_location=device).eval()

    # ---- 前処理（学習時と同一の scaler / align）----
    t3_std, m9_std, y_std, y_mean, y_std_vec = load_std_data_for_diagnosis(datapath, selector_dir)

    # ---- 診断 ----
    quick_diagnose(mdn, selector, t3_std, m9_std, y_std)
