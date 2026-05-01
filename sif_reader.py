import sif_parser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from scipy.ndimage import map_coordinates

# ── Load ──────────────────────────────────────────────────────────────────────
data, info = sif_parser.np_open('07232_TS_shotn.sif')
frame = data[0] if data.ndim == 3 else np.squeeze(data)
H, W = frame.shape

# ── Layout — all axes fixed, never recreated ──────────────────────────────────
fig = plt.figure(figsize=(16, 8))

# Fixed positions — nothing ever moves these
ax_img  = fig.add_axes([0.04, 0.18, 0.40, 0.76])   # image
ax_cbar = fig.add_axes([0.45, 0.18, 0.01, 0.76])   # colorbar (dedicated, not attached to ax_img)
ax_h    = fig.add_axes([0.50, 0.18, 0.44, 0.76])   # horizontal lineout (always present)
ax_v    = fig.add_axes([0.83, 0.18, 0.11, 0.76])   # vertical lineout (point mode only)
ax_vmin = fig.add_axes([0.04, 0.08, 0.18, 0.03])
ax_vmax = fig.add_axes([0.04, 0.02, 0.18, 0.03])

im = ax_img.imshow(frame, cmap='inferno', origin='lower',
                   vmin=frame.min(), vmax=frame.max(), aspect='auto')

# Colorbar in its own axes — ax_img is never touched again
fig.colorbar(im, cax=ax_cbar, label='Intensity')

ax_img.set_title('L-click: crosshair  |  L-drag: line  |  R-drag: box  |  Drag to move/resize', fontsize=9)

ax_v.set_visible(False)  # hidden until point mode

s_vmin = Slider(ax_vmin, 'vmin', frame.min(), frame.max(), valinit=frame.min())
s_vmax = Slider(ax_vmax, 'vmax', frame.min(), frame.max(), valinit=frame.max())

def set_lineout_mode(mode):
    """Resize ax_h and toggle ax_v — no axes created or destroyed."""
    if mode == 'cross':
        ax_h.set_position([0.50, 0.18, 0.32, 0.76])
        ax_v.set_position([0.83, 0.18, 0.11, 0.76])
        ax_v.set_visible(True)
    else:
        ax_h.set_position([0.50, 0.18, 0.44, 0.76])
        ax_v.set_visible(False)

# ── State ─────────────────────────────────────────────────────────────────────
rois = []

st = dict(
    drawing=False,
    ready_to_draw=False,
    drag_roi=None,
    drag_node=None,
    translating=False,
    tx0=None, ty0=None,
    tp1=None, tp2=None,
    press_x=None, press_y=None,
    press_btn=None,
)

DRAG_THRESH  = 4
NODE_RADIUS  = 9
SHAPE_RADIUS = 7

# ── Lineout helpers ───────────────────────────────────────────────────────────
def line_profile(p1, p2, n=600):
    r0,c0 = p1[1],p1[0];  r1,c1 = p2[1],p2[0]
    rows = np.linspace(r0,r1,n);  cols = np.linspace(c0,c1,n)
    dist = np.sqrt((r1-r0)**2+(c1-c0)**2)*np.linspace(0,1,n)
    return dist, map_coordinates(frame,[rows,cols],order=1)

def box_profile(p1, p2):
    c0,c1 = sorted([int(p1[0]),int(p2[0])])
    r0,r1 = sorted([int(p1[1]),int(p2[1])])
    r0,r1 = np.clip([r0,r1],0,H-1);  c0,c1 = np.clip([c0,c1],0,W-1)
    region = frame[r0:r1+1, c0:c1+1]
    return np.arange(c0,c0+region.shape[1]), region.sum(axis=0)

def point_profiles(px, py):
    col = int(np.clip(px,0,W-1));  row = int(np.clip(py,0,H-1))
    return np.arange(W), frame[row,:], np.arange(H), frame[:,col]

# ── Hit tests ─────────────────────────────────────────────────────────────────
def disp_pt(pt):
    return ax_img.transData.transform(pt)

def display_dist(x1,y1,x2,y2):
    a,b = disp_pt((x1,y1)), disp_pt((x2,y2))
    return np.hypot(*(a-b))

def near_node(x, y, roi):
    for name in ('p1','p2'):
        if name in roi and display_dist(x,y,*roi[name]) < NODE_RADIUS:
            return name
    return None

def on_line(x, y, roi):
    A,B,P = disp_pt(roi['p1']),disp_pt(roi['p2']),disp_pt((x,y))
    AB = B-A;  t = np.clip(np.dot(P-A,AB)/(np.dot(AB,AB)+1e-12),0,1)
    return np.hypot(*(P-(A+t*AB))) < SHAPE_RADIUS

def in_box(x, y, roi):
    p1,p2 = roi['p1'],roi['p2']
    return (min(p1[0],p2[0]) <= x <= max(p1[0],p2[0]) and
            min(p1[1],p2[1]) <= y <= max(p1[1],p2[1]))

def on_point(x, y, roi):
    return display_dist(x,y,*roi['p1']) < NODE_RADIUS

def hit_test(x, y):
    for i,roi in enumerate(rois):
        t = roi['type']
        if t == 'line':
            nd = near_node(x,y,roi)
            if nd: return i, nd
            if on_line(x,y,roi): return i, None
        elif t == 'box':
            nd = near_node(x,y,roi)
            if nd: return i, nd
            if in_box(x,y,roi): return i, None
        elif t == 'point':
            if on_point(x,y,roi): return i, 'p1'
    return None, None

# ── ROI drawing ───────────────────────────────────────────────────────────────
def clear_roi_artists(roi):
    for a in roi.get('artists',[]):
        try: a.remove()
        except: pass
    roi['artists'] = []

def remove_all_rois():
    global rois
    for roi in rois:
        clear_roi_artists(roi)
    rois = []

def draw_roi(roi):
    clear_roi_artists(roi)
    arts = []
    t = roi['type'];  p1 = roi['p1']
    p2 = roi.get('p2', p1)

    if t == 'line':
        ln, = ax_img.plot([p1[0],p2[0]],[p1[1],p2[1]],
                          color='cyan',lw=1.5,ls='--',zorder=3)
        arts.append(ln)
        for pt in [p1,p2]:
            d, = ax_img.plot(*pt,'o',color='white',ms=8,mew=1.5,
                             markeredgecolor='black',zorder=5)
            arts.append(d)

    elif t == 'box':
        x0,y0 = min(p1[0],p2[0]),min(p1[1],p2[1])
        w,h   = abs(p2[0]-p1[0]),abs(p2[1]-p1[1])
        rect  = patches.Rectangle((x0,y0),w,h,lw=1.5,
                 edgecolor='lime',facecolor='none',ls='--',zorder=3)
        ax_img.add_patch(rect); arts.append(rect)
        for pt in [p1,p2]:
            d, = ax_img.plot(*pt,'o',color='white',ms=8,mew=1.5,
                             markeredgecolor='black',zorder=5)
            arts.append(d)

    elif t == 'point':
        px,py = p1
        hl, = ax_img.plot([0,W],[py,py],color='yellow',lw=1,ls=':',zorder=3)
        vl, = ax_img.plot([px,px],[0,H],color='yellow',lw=1,ls=':',zorder=3)
        dot, = ax_img.plot(px,py,'o',color='yellow',ms=7,mew=1.5,
                           markeredgecolor='black',zorder=5)
        arts += [hl,vl,dot]

    roi['artists'] = arts

# ── Lineout plotting ──────────────────────────────────────────────────────────
def update_lineouts():
    ax_h.cla(); ax_v.cla()

    if not rois:
        set_lineout_mode('single')
        ax_h.set_title('Lineout')
        fig.canvas.draw_idle()
        return

    roi = rois[0]
    t   = roi['type']

    if t == 'line':
        set_lineout_mode('single')
        dist,prof = line_profile(roi['p1'],roi['p2'])
        ax_h.plot(dist,prof,color='cyan',lw=1.5)
        ax_h.set_xlabel('Distance (px)'); ax_h.set_ylabel('Intensity')
        ax_h.set_title(f"Line  ({roi['p1'][0]:.0f},{roi['p1'][1]:.0f})"
                       f"→({roi['p2'][0]:.0f},{roi['p2'][1]:.0f})")
        ax_h.grid(True,alpha=0.3)

    elif t == 'box':
        set_lineout_mode('single')
        xs,prof = box_profile(roi['p1'],roi['p2'])
        ax_h.plot(xs,prof,color='lime',lw=1.5)
        ax_h.fill_between(xs,prof,alpha=0.2,color='lime')
        ax_h.set_xlabel('Column (px)'); ax_h.set_ylabel('Summed Intensity')
        ax_h.set_title('Box — horizontal sum')
        ax_h.grid(True,alpha=0.3)

    elif t == 'point':
        set_lineout_mode('cross')
        px,py = roi['p1']
        xs,hprof,ys,vprof = point_profiles(px,py)

        ax_h.plot(xs,hprof,color='yellow',lw=1.5)
        ax_h.axvline(px,color='white',lw=1,ls=':',alpha=0.6)
        ax_h.set_xlabel('Column (px)'); ax_h.set_ylabel('Intensity')
        ax_h.set_title(f'Horizontal  y={py:.0f}')
        ax_h.grid(True,alpha=0.3)

        ax_v.plot(vprof,ys,color='orange',lw=1.5)
        ax_v.axhline(py,color='white',lw=1,ls=':',alpha=0.6)
        ax_v.set_xlabel('Intensity')
        ax_v.set_ylabel('Row (px)')
        ax_v.yaxis.set_label_position('right')
        ax_v.yaxis.tick_right()
        ax_v.set_title(f'Vertical  x={px:.0f}')
        ax_v.grid(True,alpha=0.3)
        ax_v.set_ylim(0,H)

    fig.canvas.draw_idle()

# ── Mouse events ──────────────────────────────────────────────────────────────
def on_press(event):
    if event.inaxes != ax_img: return
    x,y = event.xdata,event.ydata
    btn = event.button

    st['press_x']=x; st['press_y']=y; st['press_btn']=btn
    st['drawing']=False; st['ready_to_draw']=False
    st['drag_roi']=None; st['drag_node']=None; st['translating']=False

    idx,node = hit_test(x,y)
    if idx is not None:
        roi = rois[idx]
        if node:
            st['drag_roi']=idx; st['drag_node']=node
        else:
            st['drag_roi']=idx; st['translating']=True
            st['tx0'],st['ty0']=x,y
            st['tp1']=roi['p1']
            st['tp2']=roi.get('p2',roi['p1'])
    else:
        st['ready_to_draw']=True

def on_motion(event):
    if event.inaxes != ax_img: return
    x,y = event.xdata,event.ydata
    px,py = st['press_x'],st['press_y']
    if px is None: return

    moved = display_dist(px,py,x,y) > DRAG_THRESH

    if st['drag_node'] is not None:
        roi = rois[st['drag_roi']]
        roi[st['drag_node']] = (x,y)
        draw_roi(roi); update_lineouts(); return

    if st['translating']:
        roi = rois[st['drag_roi']]
        dx,dy = x-st['tx0'], y-st['ty0']
        roi['p1'] = (st['tp1'][0]+dx, st['tp1'][1]+dy)
        if 'p2' in roi:
            roi['p2'] = (st['tp2'][0]+dx, st['tp2'][1]+dy)
        draw_roi(roi); update_lineouts(); return

    if moved and st['ready_to_draw']:
        dtype = 'line' if st['press_btn']==1 else 'box'
        if not st['drawing']:
            remove_all_rois()
            new_roi = dict(type=dtype, p1=(px,py), p2=(x,y), artists=[])
            rois.append(new_roi)
            st['drawing']=True
            st['drag_roi']=len(rois)-1
        else:
            rois[st['drag_roi']]['p2'] = (x,y)
        draw_roi(rois[st['drag_roi']]); update_lineouts()

def on_release(event):
    if event.inaxes != ax_img: return
    x,y = event.xdata,event.ydata
    px,py = st['press_x'],st['press_y']
    if px is None: return

    moved = display_dist(px,py,x,y) > DRAG_THRESH

    if not moved and event.button==1 and st['ready_to_draw']:
        remove_all_rois()
        new_roi = dict(type='point', p1=(x,y), artists=[])
        rois.append(new_roi)
        draw_roi(new_roi); update_lineouts()

    st['drawing']=False; st['ready_to_draw']=False
    st['drag_roi']=None; st['drag_node']=None
    st['translating']=False; st['press_x']=st['press_y']=None

def on_clim(val):
    im.set_clim(s_vmin.val, s_vmax.val); fig.canvas.draw_idle()

s_vmin.on_changed(on_clim); s_vmax.on_changed(on_clim)
fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('motion_notify_event',  on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.suptitle('SIF Viewer', fontsize=12)
plt.show()
