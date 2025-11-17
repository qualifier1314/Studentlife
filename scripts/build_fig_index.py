import os

"""
生成 outputs/static/figures/index.html，展示目录下的 PNG/JPG 图像，
以网格方式直接预览，无需点击文件名。
"""


def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def build_index():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fig_dir = os.path.join(base_dir, 'outputs', 'static', 'figures')
    ensure_dir(fig_dir)
    files = [f for f in os.listdir(fig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # 排序：优先新的无“静态”前缀的文件
    files.sort(key=lambda x: (x.startswith('静态'), x))

    html = """
<!doctype html>
<html lang=zh>
<head>
  <meta charset="utf-8">
  <title>三层网络图预览</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, Segoe UI, PingFang SC, Microsoft YaHei, Arial, sans-serif; margin: 16px; }
    h1 { font-size: 20px; margin: 8px 0 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr)); grid-gap: 16px; }
    .card { border: 1px solid #e5e5e5; border-radius: 8px; overflow: hidden; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    .card img { width: 100%; height: auto; display: block; }
    .meta { padding: 8px 10px; font-size: 14px; color: #333; display:flex; align-items:center; justify-content:space-between; }
    .meta a { text-decoration: none; color: #0a66c2; }
  </style>
  </head>
<body>
  <h1>输出图像预览（点击文件名下载原图）</h1>
  <div class="grid">
"""

    for f in files:
        html += f"  <div class=\"card\">\n    <img src=\"{f}\" alt=\"{f}\"/>\n    <div class=\"meta\"><span>{f}</span><a href=\"{f}\" download>下载</a></div>\n  </div>\n"

    html += """
  </div>
</body>
</html>
"""

    out_path = os.path.join(fig_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Build] index.html 已生成 -> {out_path}")


if __name__ == '__main__':
    build_index()