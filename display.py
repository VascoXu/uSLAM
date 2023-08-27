import os
import sdl2
import sdl2.ext

class Display2D(object):
    def __init__(self, W, H):
        sdl2.ext.init()

        self.W, self.H = H, W
        self.window = sdl2.ext.Window('slam', size=(W, H), position=(-500, -500))

        self.window.show()

    def paint(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0, 1)
        self.window.refresh()
    

from multiprocessing import Process, Queue
import pypangolin as pango
import OpenGL.GL as gl
import numpy as np


class Display3D(object):
    def __init__(self):
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()
        
    def viewer_thread(self, q):
        self.viewer_init(1024, 768)
        while 1:
            self.viewer_refresh(q)
            
    def viewer_init(self, w, h):
        pango.CreateWindowAndBind('Main', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)
        
        pm = pango.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000)
        mv = pango.ModelViewLookAt(0, -10, -8,
                                   0, 0, 0,
                                   0, -1, 0)
        self.s_cam = pango.OpenGlRenderState(pm, mv)

        self.handler = pango.Handler3D(self.s_cam)
        self.d_cam = (
                pango.CreateDisplay()
                .SetBounds(
                    pango.Attach(0),
                    pango.Attach(1),
                    pango.Attach(0),
                    pango.Attach(1),
                    w/h,
                )
                .SetHandler(self.handler)
        )
        self.d_cam.Activate()

    def viewer_refresh(self, q):
        while not q.empty():
            self.state = q.get()

        # turn state into points
        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])[:, :3]

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        self.d_cam.Activate(self.s_cam)
        
        if self.state is not None:
            if self.state[0].shape[0] >= 2:
                # draw poses
                gl.glPointSize(10)
                gl.glColor3f(0.0, 1.0, 0.0)
                pango.glDrawPoints(ppts[:-1])

            if self.state[0].shape[0] >= 1:
                # draw current pose as yellow
                gl.glPointSize(10)
                gl.glColor3f(1.0, 1.0, 0.0)
                pango.glDrawPoints(ppts[-1:])

            # draw keypoints
            if self.state[1].shape[0] != 0:
                if os.getenv("COLOR"):
                    # draw points with original color
                    for i in range(len(spts)):
                        gl.glPointSize(3)
                        gl.glColor3f(*self.state[2][i])
                        pango.glDrawPoints([spts[i]])
                else:
                    # draw points in white
                    gl.glPointSize(3)
                    gl.glColor3f(1.0, 1.0, 1.0)
                    pango.glDrawPoints(spts)

        pango.FinishFrame()

    def paint(self, mapp):
        if self.q is None:
            return

        poses, pts, colors = [], [], []
        for f in mapp.frames:
            # invert poses for display only
            poses.append(np.linalg.inv(f.pose))
        for p in mapp.points:
            pts.append(p.pt)
            colors.append(p.color)
        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))
