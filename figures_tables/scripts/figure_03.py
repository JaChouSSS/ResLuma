
from pathlib import Path
from itertools import product
import math, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib import ticker
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe

# ============================================================
# 字体包裹宽度设置：以后只需要修改这里
# ============================================================

# X、Y轴刻度数字周围的黑色描边宽度
# 0 = 不描边，0.1 = 很细，0.3 = 较明显，0.5 = 更粗
TICK_STROKE_WIDTH = 0.000

# 方法名称周围白色背景的留白宽度
# 原始值为0.06，数值越大，白色背景越宽
LABEL_BG_PAD = 0.06

# ============================================================

BASE = Path(__file__).resolve().parent
FIGDIR = BASE.parent / 'generated_figures'

# Authoritative values taken from existing,
# paper Table 1 and 4K GPU timing table.
data = [
    ('ResLuma', 28, 19.332, .4539, 8.45),
    ('LiteIE', 58, 19.203, .4321, 14.38),
    ('SCI', 258, 14.853, .4158, 9.55),
    ('RUAS', 3438, 16.448, .2792, 123.77),
    ('MobileIE', 4075, 23.634, .3528, 25.24),
    ('Zero-DCE++', 10561, 15.355, .4742, 11.64),
    ('Zero-DCE', 79416, 14.861, .4780, 84.80),
    ('FourLLIE', 119446, 20.023, .4900, 276.27),
    ('URetinex-Net', 340105, 19.861, .5075, 646.34),
    ('Retinexformer', 1605701, 25.155, .4055, 1453.59),
]

assert len(data) == len({x[0] for x in data}) == 10

methods = [
    dict(name=n, params=p, psnr=s, clip=k, ms=t)
    for n, p, s, k, t in data
]


def pareto(q, c):
    return sorted(
        [
            m for m in methods
            if not any(
                (x[c] <= m[c] and x[q] >= m[q])
                and (x[c] < m[c] or x[q] > m[q])
                for x in methods if x is not m
            )
        ],
        key=lambda x: x[c]
    )


# ============================================================
# 全局字体及绘图样式
# ============================================================

rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,

    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.15,

    'axes.labelsize': 11.8,
    'xtick.labelsize': 10.2,
    'ytick.labelsize': 10.2,
    'axes.titlesize': 11.6,

    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',

    # 对数坐标使用Liberation Sans数学字体
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold',
    'mathtext.default': 'regular',
})

navy = '#0B47B8'
charcoal = '#000000'
gray = '#3F4A54'
orange = '#FF2323'
grid = '#D8E0E6'

fig, axes = plt.subplots(
    2, 2,
    figsize=(7.65, 6.05),
    dpi=220
)

fig.subplots_adjust(
    left=.095,
    right=.975,
    top=.932,
    bottom=.128,
    hspace=.46,
    wspace=.275
)

configs = [
    (0, 0, 'psnr', 'params',
     '(a) LOL PSNR vs. parameters'),
    (0, 1, 'psnr', 'ms',
     '(b) LOL PSNR vs. 4K latency'),
    (1, 0, 'clip', 'params',
     '(c) BIMEF CLIP-IQA vs. parameters'),
    (1, 1, 'clip', 'ms',
     '(d) BIMEF CLIP-IQA vs. 4K latency'),
]


# ============================================================
# 绘制四个子图
# ============================================================

for r, k, q, c, title in configs:
    ax = axes[r, k]

    ax.set_xscale('log')
    ax.set_axisbelow(True)

    ax.grid(
        which='major',
        axis='both',
        color=grid,
        lw=.55
    )

    ax.set_title(
        title,
        loc='left',
        fontweight='semibold',
        pad=9
    )

    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    ax.spines['left'].set_linewidth(1.15)
    ax.spines['bottom'].set_linewidth(1.15)

    ax.tick_params(
        direction='out',
        length=3.2,
        width=1.0,
        color='black',
        labelcolor='black'
    )

    if c == 'params':
        ax.set_xlim(13, 5e6)
        ax.set_xlabel('Model parameters (log scale)')

        ax.xaxis.set_major_locator(
            ticker.LogLocator(
                base=10,
                subs=(1,),
                numticks=7
            )
        )
    else:
        ax.set_xlim(5, 2600)
        ax.set_xlabel('4K inference latency (ms, log scale)')

        ax.xaxis.set_major_locator(
            ticker.LogLocator(
                base=10,
                subs=(1,),
                numticks=5
            )
        )

    ax.xaxis.set_minor_locator(ticker.NullLocator())

    if q == 'psnr':
        ax.set_ylim(12.95, 27.5)
        ax.set_yticks([15, 20, 25])

        if k == 0:
            ax.set_ylabel('LOL PSNR (dB)')

    else:
        ax.set_ylim(.24, .554)
        ax.set_yticks([.3, .4, .5])

        if k == 0:
            ax.set_ylabel('BIMEF CLIP-IQA')

    front = pareto(q, c)
    front_names = {x['name'] for x in front}

    ax.plot(
        [x[c] for x in front],
        [x[q] for x in front],
        color=navy,
        lw=1.6,
        linestyle=(0, (4, 2.3)),
        alpha=1.0,
        zorder=2
    )

    for m in methods:

        if m['name'] == 'ResLuma':
            ax.scatter(
                m[c], m[q],
                color=orange,
                edgecolor='#A81212',
                linewidth=.7,
                s=130,
                marker='*',
                zorder=5
            )

        elif m['name'] in front_names:
            ax.scatter(
                m[c], m[q],
                color=navy,
                edgecolor='white',
                linewidth=.55,
                s=34,
                marker='D',
                zorder=4
            )

        else:
            ax.scatter(
                m[c], m[q],
                color=gray,
                edgecolor='white',
                linewidth=.5,
                s=28,
                marker='o',
                zorder=3
            )


# ============================================================
# X、Y轴刻度数字：字体与描边设置
# ============================================================

# renderer和dpi必须保留，后面的标签避让算法需要
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
dpi = fig.dpi

for ax in axes.flat:
    for lab in ax.get_xticklabels() + ax.get_yticklabels():

        # 刻度数字的字体
        lab.set_fontfamily('Liberation Sans')
        lab.set_fontweight('normal')

        # 字体周围黑色描边，宽度由开头参数控制
        if TICK_STROKE_WIDTH > 0:
            lab.set_path_effects([
                pe.withStroke(
                    linewidth=TICK_STROKE_WIDTH,
                    foreground='black'
                )
            ])
        else:
            lab.set_path_effects([])


# ============================================================
# 标签候选位置
# ============================================================

random.seed(17)

DIRECTIONS = [
    (8, 0), (-8, 0),
    (8, 10), (-8, 10),
    (8, -10), (-8, -10),
    (8, 18), (-8, 18),
    (8, -18), (-8, -18),
    (16, 0), (-16, 0),
    (18, 12), (-18, 12),
    (18, -12), (-18, -12),
    (6, 25), (-6, 25),
    (6, -25), (-6, -25),
    (30, 0), (-30, 0),
    (29, 14), (-29, 14),
    (29, -14), (-29, -14),
    (18, 23), (-18, 23),
    (18, -23), (-18, -23),
    (5, 35), (-5, 35),
    (5, -35), (-5, -35),
]


# ============================================================
# Pareto标签优先位置
# ============================================================

PREF = {
    ('psnr', 'params'): {
        'ResLuma': (8, 12),
        'MobileIE': (8, 10),
        'Retinexformer': (-8, -12)
    },

    ('psnr', 'ms'): {
        'ResLuma': (8, 12),
        'MobileIE': (8, 10),
        'Retinexformer': (-8, -12)
    },

    ('clip', 'params'): {
        'ResLuma': (8, 12),
        'Zero-DCE++': (8, 12),
        'Zero-DCE': (-8, -12),
        'FourLLIE': (-8, 12),
        'URetinex-Net': (-8, 11)
    },

    ('clip', 'ms'): {
        'ResLuma': (8, 12),
        'Zero-DCE++': (8, 10),
        'Zero-DCE': (-8, -12),
        'FourLLIE': (-8, 12),
        'URetinex-Net': (-8, 11)
    },
}


# ============================================================
# 自动标签布局与碰撞检测
# ============================================================

STATS = {}

for r, k, q, c, title in configs:

    ax = axes[r, k]
    bbox = ax.bbox

    front_names = {
        x['name'] for x in pareto(q, c)
    }

    pts = {
        m['name']: ax.transData.transform((m[c], m[q]))
        for m in methods
    }

    fs = {
        m['name']: (
            9.8 if m['name'] == 'ResLuma'
            else 9.2 if m['name'] in front_names
            else 8.7
        )
        for m in methods
    }

    sizes = {}

    for m in methods:
        name = m['name']

        prop = FontProperties(
            family='DejaVu Sans',
            size=fs[name],
            weight='bold' if name in front_names else 'normal'
        )

        w, h, desc = (
            renderer.get_text_width_height_descent(
                name,
                prop,
                ismath=False
            )
        )

        sizes[name] = (w + 2, h + 2)

    # Bounding boxes in pixels
    def coord_rect(name, idx):
        dx, dy = DIRECTIONS[idx]

        x, y = pts[name]
        w, h = sizes[name]

        ox = dx * dpi / 72
        oy = dy * dpi / 72

        xa = x + ox
        ya = y + oy

        x1 = xa if dx >= 0 else xa - w
        y1 = ya - h / 2

        return (x1, y1, x1 + w, y1 + h)

    def overlap(A, B, pad=1):
        return (
            max(
                0,
                min(A[2] + pad, B[2] + pad)
                - max(A[0] - pad, B[0] - pad)
            )
            *
            max(
                0,
                min(A[3] + pad, B[3] + pad)
                - max(A[1] - pad, B[1] - pad)
            )
        )

    def loc_cost(name, idx, assigned):

        rect = coord_rect(name, idx)
        dx, dy = DIRECTIONS[idx]

        l, b, ri, t = rect

        x0 = bbox.x0 + 3
        y0 = bbox.y0 + 5
        x1 = bbox.x1 - 3
        y1 = bbox.y1 - 3

        out = (
            max(0, x0 - l)
            + max(0, ri - x1)
            + max(0, y0 - b)
            + max(0, t - y1)
        )

        cost = 80 * out * out

        for other, oi in assigned.items():
            if other == name:
                continue

            cost += 50 * overlap(
                rect,
                coord_rect(other, oi),
                pad=2
            )

        for other, (px, py) in pts.items():
            if other == name:
                continue

            if (
                rect[0] - 4 < px < rect[2] + 4
                and rect[1] - 4 < py < rect[3] + 4
            ):
                cost += 800

        cost += .35 * (
            math.hypot(dx, dy) - 11
        ) ** 2

        cost += 1.5 * abs(dy)

        if name in front_names:
            cost += .65 * (
                math.hypot(dx, dy) - 9
            ) ** 2

        pref = PREF.get((q, c), {}).get(name)

        if pref:
            cost += .30 * (
                (dx - pref[0]) ** 2
                + (dy - pref[1]) ** 2
            )

        return cost

    # Pareto frontier methods first
    order = (
        [
            m['name'] for m in methods
            if m['name'] in front_names
        ]
        +
        [
            m['name'] for m in methods
            if m['name'] not in front_names
        ]
    )

    assigned = {}

    for name in order:
        assigned[name] = min(
            range(len(DIRECTIONS)),
            key=lambda ix: loc_cost(
                name,
                ix,
                assigned
            )
        )

    # Coordinate descent to avoid label collisions
    for iteration in range(18):
        changed = False

        for name in (
            reversed(order)
            if iteration % 2 == 0
            else order
        ):
            opt = min(
                range(len(DIRECTIONS)),
                key=lambda ix: loc_cost(
                    name,
                    ix,
                    assigned
                )
            )

            if opt != assigned[name]:
                assigned[name] = opt
                changed = True

        if not changed:
            break

    collisions = []

    for i, n in enumerate(order):
        for m in order[i + 1:]:
            o = overlap(
                coord_rect(n, assigned[n]),
                coord_rect(m, assigned[m]),
                pad=1
            )

            if o > 0:
                collisions.append(
                    (n, m, round(o, 1))
                )

    print(
        title,
        'label overlaps:',
        collisions
    )

    STATS[(q, c)] = collisions


    # ========================================================
    # 绘制方法名称
    # ========================================================

    for m in methods:
        name = m['name']

        dx, dy = DIRECTIONS[assigned[name]]
        infront = name in front_names

        kw = dict(
            fontsize=fs[name],

            ha='left' if dx > 0 else 'right',
            va='center',

            color=(
                orange if name == 'ResLuma'
                else navy if infront
                else charcoal
            ),

            fontweight=(
                'bold' if infront else 'normal'
            ),

            # 白色背景包裹
            # 宽度由LABEL_BG_PAD控制
            bbox=dict(
                boxstyle=f'square,pad={LABEL_BG_PAD}',
                fc='white',
                ec='none',
                alpha=.92
            ),

            clip_on=False,
            zorder=8
        )

        # Subtle connectors for off-center labels only
        if abs(dx) > 14 or abs(dy) > 12:
            kw['arrowprops'] = dict(
                arrowstyle='-',
                lw=.6,
                color='#78848E',
                shrinkA=1.5,
                shrinkB=3
            )

        ax.annotate(
            name,
            (m[c], m[q]),
            xytext=(dx, dy),
            textcoords='offset points',
            **kw
        )


# ============================================================
# 图例
# ============================================================

handles = [
    Line2D(
        [], [],
        marker='*',
        linestyle='none',
        markerfacecolor=orange,
        markeredgecolor='#A81212',
        markersize=10.5,
        label='ResLuma'
    ),

    Line2D(
        [], [],
        marker='D',
        linestyle='none',
        markerfacecolor=navy,
        markeredgecolor='white',
        markersize=6.8,
        label='Other Pareto-optimal methods'
    ),

    Line2D(
        [], [],
        marker='o',
        linestyle='none',
        markerfacecolor=gray,
        markeredgecolor='white',
        markersize=6.4,
        label='Dominated methods'
    ),

    Line2D(
        [], [],
        color=navy,
        linestyle=(0, (4, 2.3)),
        lw=1.5,
        label='Pareto frontier'
    ),
]

fig.legend(
    handles=handles,
    loc='lower center',
    bbox_to_anchor=(.50, .005),
    ncol=4,
    frameon=False,
    fontsize=9.4,
    columnspacing=1.2,
    handletextpad=.4
)


# ============================================================
# 导出PDF与PNG
# ============================================================

FIGDIR.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(
    FIGDIR / 'Figure_03.pdf',
    format='pdf'
)

fig.savefig(
    FIGDIR / 'Figure_03.png',
    dpi=180
)

plt.close(fig)

assert all(
    not v for v in STATS.values()
), f'Label overlaps: {STATS}'

print(
    'Wrote vector figure',
    FIGDIR / 'Figure_03.pdf'
)
