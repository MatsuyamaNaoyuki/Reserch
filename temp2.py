import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from myclass import myfunction, Mydataset, MyModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========= 既存ユーティリティ（そのまま利用） =========
@torch.no_grad()
def teacher_label_and_margin(mu_std, y_std):
    d1 = torch.norm(mu_std[:,0,:] - y_std, dim=1)
    d2 = torch.norm(mu_std[:,1,:] - y_std, dim=1)
    teacher = (d2 < d1).long()           # 0: mu1が近い, 1: mu2が近い
    margin  = (d1 - d2).abs()
    return teacher, margin, d1, d2

def make_margin_weights(margin, low_q=0.2, high_q=0.8):
    lo = torch.quantile(margin, low_q)
    hi = torch.quantile(margin, high_q)
    w = (margin - lo) / (hi - lo + 1e-8)
    w = torch.clamp(w, 0.0, 1.0)
    return w, lo, hi

def get_class_weight(teacher):
    num0 = (teacher==0).sum().float()
    num1 = (teacher==1).sum().float()
    w0 = (num0 + num1) / (2.0 * num0 + 1e-8)
    w1 = (num0 + num1) / (2.0 * num1 + 1e-8)
    return torch.tensor([w0, w1], device=teacher.device)

# ========= 追加: 磁気の時系列を切り出す =========
def build_mag_sequences(mag9_std, rot3_std, y_std, typedf, L=16, stride=1):
    """
    typedf: 1 trial の末尾インデックス（昇順）を仮定
    各 trial 内で過去Lフレームを因果窓で切り出し:
      seq_mag: [Nseq, L, 9]
      mu/y は窓の“現在”=末尾フレームに合わせて作るため、その時点の index 配列 js も返す
    """
    typedf = typedf.tolist()
    typedf.insert(0, 0)
    total_span = (L - 1) * stride

    seq_mag, js_all = [], []
    # rot3_std, y_std は“現在”用のインデックス js で拾うのでここでは触らない

    for i in range(len(typedf)-1):
        start = typedf[i] + total_span
        end   = typedf[i+1]
        if end <= start:
            continue

        js = torch.arange(start, end, device=mag9_std.device)  # 現在時刻の位置
        # 過去Lフレームのインデックスを作る
        rel = torch.arange(L-1, -1, -1, device=mag9_std.device) * stride
        idx = js.unsqueeze(1) - rel    # [num, L]

        # ここで NaN を含む行を除外（必要なら）
        # 今は簡単に実装: そのまま使う
        seq_mag.append(mag9_std[idx])  # [num, L, 9]
        js_all.append(js)

    seq_mag = torch.cat(seq_mag, dim=0) if len(seq_mag)>0 else torch.empty(0, L, 9, device=mag9_std.device)
    js_all  = torch.cat(js_all, dim=0)  if len(js_all)>0  else torch.empty(0, device=mag9_std.device, dtype=torch.long)
    # rot3_std/js_all からMDNを呼び、y_std/js_all を教師に使うため、js_allも返す
    return seq_mag, js_all

# ========= モデル: 磁気時系列→GRU→特徴、mu1/mu2と結合して2クラス分類 =========
class SelectorGRU(nn.Module):
    def __init__(self, mag_dim=9, hidden=64, num_layers=1, fc_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_size=mag_dim, hidden_size=hidden, num_layers=num_layers,
                          batch_first=True, bidirectional=False)
        # 出力: h_T (B, hidden)
        # mu1/mu2 は3次元×2個=6次元 → 合計 hidden+6 を最終MLPへ
        self.mlp = nn.Sequential(
            nn.Linear(hidden + 6, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 2)  # 2クラス
        )

    def forward(self, mag_seq, mu_pair_flat):
        """
        mag_seq: (B, L, 9)
        mu_pair_flat: (B, 6) = [mu1(3), mu2(3)]（MDN出力の“現在フレーム”）
        """
        out, h = self.gru(mag_seq)   # h: (1, B, hidden)
        h_last = h[-1]               # (B, hidden)
        x = torch.cat([h_last, mu_pair_flat], dim=1)  # (B, hidden+6)
        logits = self.mlp(x)
        return logits

# ========= 学習ループ =========
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for mag_seq, mu_pair, label, w in loader:
        mag_seq = mag_seq.to(device)          # (B, L, 9)
        mu_pair = mu_pair.to(device)          # (B, 6)
        label   = label.to(device)            # (B,)
        w       = w.to(device)                # (B,)

        logits = model(mag_seq, mu_pair)
        loss_each = criterion(logits, label)  # (B,)
        loss = (loss_each * w).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            total_correct += (pred == label).sum().item()
            total_loss += loss.item() * mag_seq.size(0)
            total_count += mag_seq.size(0)

    return total_loss/total_count, total_correct/total_count

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for mag_seq, mu_pair, label, w in loader:
        mag_seq = mag_seq.to(device)
        mu_pair = mu_pair.to(device)
        label   = label.to(device)
        w       = w.to(device)

        logits = model(mag_seq, mu_pair)
        loss_each = criterion(logits, label)
        loss = (loss_each * w).mean()

        pred = logits.argmax(dim=1)
        total_correct += (pred == label).sum().item()
        total_loss += loss.item() * mag_seq.size(0)
        total_count += mag_seq.size(0)

    return total_loss/total_count, total_correct/total_count

# ========= メイン =========
def main():
    # ---- 設定 ----
    L = 16
    stride = 1
    batch_size = 128
    num_epochs = 200
    train_ratio = 0.8
    patience_stop = 30
    patience_sched = 10

    result_dir = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\select_gru_magL16"
    os.makedirs(result_dir, exist_ok=True)

    # ---- モデル類ロード ----
    mdn_path = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\MDN2\model.pth"
    MDN = torch.jit.load(mdn_path, map_location=device).eval()

    # ---- データ読み込み（学習用大規模ファイル）----
    train_pickle = r"C:\Users\WRS\Desktop\Matsuyama\laerningdataandresult\reretubefinger0819\mixhit1500kaifortrain.pickle"
    x_data, y_data, typedf = myfunction.read_pickle_to_torch(train_pickle, motor_angle=True, motor_force=True, magsensor=True)
    y_last3 = y_data[:, -3:]  # (N, 3)

    # スケーリング（あなたの既存手順に合わせる）
    base_dir = os.path.dirname(result_dir)
    kijun_path = myfunction.find_pickle_files("kijun", base_dir)
    kijun_x, _, _ = myfunction.read_pickle_to_torch(kijun_path, motor_angle=True, motor_force=True, magsensor=True)

    x_std_all, x_mean_all, x_std_all_s, fitA = Mydataset.align_to_standardize_all(kijun_x, x_data)
    # y標準化（MDNと一致させる）
    y_mean, y_std = Mydataset.fit_standardizer_torch(y_last3)
    y_std_all = Mydataset.apply_standardize_torch(y_last3, y_mean, y_std)

    mask = torch.isfinite(x_data).all(dim=1) & torch.isfinite(y_last3).all(dim=1)
    X = x_std_all[mask]
    Y_std = y_std_all[mask]
    typedf_masked = typedf[mask.cpu()] if isinstance(typedf, torch.Tensor) else typedf  # 念のため

    # 分割
    rot3_std, force3_std, mag9_std = torch.split(X, [3, 3, 9], dim=1)

    # ---- 時系列窓の構築（磁気のみL=16）----
    mag_seq, js = build_mag_sequences(mag9_std, rot3_std, Y_std, typedf_masked, L=L, stride=stride)
    # “現在フレーム”の rot3/y を抽出
    rot_now = rot3_std[js].to(device)   # (Nseq, 3)
    y_now   = Y_std[js].to(device)      # (Nseq, 3)
    mag_seq = mag_seq.to(device)        # (Nseq, L, 9)

    # ---- MDNで mu1/mu2 を取得（現在フレームのみ）----
    with torch.no_grad():
        _, mu_std, _ = MDN(rot_now)     # (Nseq, 2, 3)

    # 教師とmargin
    label, margin, d1, d2 = teacher_label_and_margin(mu_std, y_now)
    # mu1/mu2 を平坦化（6次元）
    mu_pair = torch.cat([mu_std[:,0,:], mu_std[:,1,:]], dim=1)  # (Nseq, 6)

    # ---- 重み（margin & class weight）----
    m_w, lo, hi = make_margin_weights(margin, 0.2, 0.8)
    class_weight = get_class_weight(label)
    # CEを per-sample で出すため reduction='none'
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction='none')

    # ---- Dataset / Loader ----
    # CPUに寄せておく（DataLoaderでpin_memory活用）
    dataset = TensorDataset(mag_seq.cpu(), mu_pair.cpu(), label.long().cpu(), m_w.cpu())
    N = len(dataset)
    n_train = int(N * train_ratio)
    n_valid = N - n_train
    train_set, valid_set = random_split(dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=True, num_workers=1, persistent_workers=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1, persistent_workers=True)

    # ---- モデル/最適化 ----
    model = SelectorGRU(mag_dim=9, hidden=64, num_layers=1, fc_dim=64).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience_sched)

    # ---- 学習 ----
    best_val = float('inf'); bad = 0
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc = eval_epoch(model,  valid_loader,  criterion)
        scheduler.step(va_loss)

        pbar.set_description(f"[{epoch}] tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} tr_acc={tr_acc:.3f} va_acc={va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            # 保存
            save_path = os.path.join(result_dir, "selector_gru_magL16.pth")
            torch.jit.script(model).save(save_path)
        else:
            bad += 1
            if bad >= patience_stop:
                print(f"Early stopping @ {epoch}")
                break

    # スケーラ保存（推論時に同じ前処理ができるように）
    scaler_blob = {
        "x_mean": x_mean_all.cpu().numpy(),
        "x_std":  x_std_all_s.cpu().numpy(),
        "y_mean": y_mean.cpu().numpy(),
        "y_std":  y_std.cpu().numpy(),
        "fitA":   fitA
    }
    myfunction.wirte_pkl(scaler_blob, os.path.join(result_dir, "scaler"))

if __name__ == "__main__":
    main()
