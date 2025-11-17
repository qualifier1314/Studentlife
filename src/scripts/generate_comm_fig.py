import os
import csv
import matplotlib.pyplot as plt


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(base_dir, 'outputs', 'static')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    comm_csv = os.path.join(out_dir, 'communication_features.csv')
    fig_path = os.path.join(fig_dir, 'Communication_features.png')
    if not os.path.exists(comm_csv):
        print(f"[Skip] Not found: {comm_csv}")
        return

    try:
        with open(comm_csv, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            cols_needed = ['unique_contacts', 'comm_freq_week', 'calls_total', 'sms_total']
            if not all(c in reader.fieldnames for c in cols_needed):
                print("[Skip] Required columns missing.")
                return
            uniq, freq, size = [], [], []
            for row in reader:
                try:
                    uc = float(row.get('unique_contacts') or 0.0)
                    cw = float(row.get('comm_freq_week') or 0.0)
                    ct = float(row.get('calls_total') or 0.0)
                    st = float(row.get('sms_total') or 0.0)
                except Exception:
                    uc, cw, ct, st = 0.0, 0.0, 0.0, 0.0
                uniq.append(uc)
                freq.append(cw)
                s = max(1.0, ct + st)
                size.append(s)

        # Draw figure
        plt.figure(figsize=(10, 4))
        ax1 = plt.subplot(1, 2, 1)
        ax1.hist(uniq, bins=15, color='#4C78A8', alpha=0.85)
        ax1.set_title('Unique Contacts (hist)')
        ax1.set_xlabel('count')
        ax1.set_ylabel('students')

        ax2 = plt.subplot(1, 2, 2)
        ax2.scatter(uniq, freq, s=[10 + 2*x for x in size], c='#F58518', alpha=0.7)
        ax2.set_title('Comm Freq / week vs Unique Contacts')
        ax2.set_xlabel('unique_contacts')
        ax2.set_ylabel('comm_freq_week')
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"[Saved] {fig_path}")
    except Exception as e:
        print(f"[Error] {e}")


if __name__ == '__main__':
    main()