import numpy as np
import math

import pyglet
from pyglet.gl import *
import ctypes
from PIL import Image
import cv2


def load_processed_baseshapes(filename=""):
    if filename:
        baseshapes = np.load(filename)
    else:
        baseshapes = np.load("../../shapes/baseshapes.npy")
    return baseshapes

def load_triangles(filename=""):
    if filename:
        triangles = np.load(filename)
    else:
        triangles = np.load("../../shapes/triangles.npy")
    return triangles


def calc_shape(allshapes, e):
    assert(e.size == 46)
    # e is a list or np.array vector
    e = np.array(e).astype(np.float32)
    full_e = np.ones(e.size+1, dtype=np.float32)
    full_e[1:] = e
    shape = full_e @ allshapes
    # reshape to numVer*3
    numVert = int(shape.size / 3)
    ret_shape = np.reshape(shape, (numVert,3))
    return ret_shape


def calc_all_shapes(allshapes, Es):
    shapes = []
    for e in Es:
        shape = cal_shape(allshapes, e)
        shapes.append(shape)
    return shapes


def transform_shape(shape, R=None, T=None):
    ret_shape = np.copy(shape)
    if R is not None:
        ret_shape = ret_shape @ R.transpose()
    if T is not None:
        ret_shape = np.add(ret_shape, T)
    return ret_shape


def calc_vertex_normals(vertices, triangles):

    def normalize_v3(arr):
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens                
        return arr

    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    tris = vertices[triangles]
    n = np.cross(tris[: :,1] - tris[: :,0], tris[: :,2] - tris[: :,0])
    normalize_v3(n)
    norm[triangles[:,0]] += n
    norm[triangles[:,1]] += n
    norm[triangles[:,2]] += n
    normalize_v3(norm)
    return norm


class Renderer(object):
    def __init__(self, width=640, height=480, caption="Renderer"):
        self.win = pyglet.window.Window(width=width, height=height, caption=caption, visible=False)
        self.width = width
        self.height = height
        self.initGL()
        self.buffer = (GLubyte * (3*width*height))(0)

    def initGL(self):
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glShadeModel(GL_SMOOTH)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def render(self, shape, triangles, normals=None, text=""):
        self.win.dispatch_events()

        if normals is None:
            normals = calc_vertex_normals(shape, triangles)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glClearColor(1.0, 1.0, 1.0, 1.0)
         
        # render shape
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        ambient_light = (GLfloat*4)(*[0.25, 0.25, 0.25, 1.0])
        diffuse_light = (GLfloat*4)(*[0.7, 0.7, 0.7, 1.0])
        light_pos = (GLfloat*3)(*[-1000.0, 0.0, 200000.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient_light)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse_light)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)        

        diffuse_material = (GLfloat*4)(*[0.5, 0.5, 0.5, 1.0])
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse_material)

        # setup projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(self.width)/float(self.height), 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -4.0)

        # vertex pointer & drawarray
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        va = shape[triangles].flatten().astype(ctypes.c_float)
        no = normals[triangles].flatten().astype(ctypes.c_float)
        nTri = triangles.shape[0]
        glVertexPointer(3, GL_FLOAT, 0, va.ctypes.data)
        glNormalPointer(GL_FLOAT, 0, no.ctypes.data)
        glColor3f(0.81, 0.59, 0.49)
        glDrawArrays(GL_TRIANGLES, 0, nTri*3)

        # render the eyebrow
        #glColor3f(172.0/255, 86.0/255, 57.0/255)
        glColor3f(0.67, 0.34, 0.22)
        #left_indices = [7187, 9427, 9429, 2139, 7170, 9414]
        #right_indices = [713, 10870, 4451, 4293, 4275, 4271]
        eyebrow_tris = np.array([   [9427, 7187, 9414],
                                    [9427, 9414, 7170],
                                    [9427, 7170, 9429],
                                    [9429, 7170, 2139],
                                    [714, 4271, 10870],
                                    [10870, 4271, 4451],
                                    [4451, 4271, 4275],
                                    [4451, 4275, 4293] ], dtype=np.int32)
        ebv = shape[eyebrow_tris].flatten().astype(ctypes.c_float)
        ebn = normals[eyebrow_tris].flatten().astype(ctypes.c_float)
        glVertexPointer(3, GL_FLOAT, 0, ebv.ctypes.data)
        glNormalPointer(GL_FLOAT, 0, ebn.ctypes.data)

        glDisable(GL_DEPTH_TEST)
        glDrawArrays(GL_TRIANGLES, 0, 24)

        if text:
            #draw text
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHT0)
            glDisable(GL_LIGHTING)
            glDisable(GL_COLOR_MATERIAL)
        
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluOrtho2D(0, self.width, 0, self.height)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            pyglet.text.Label(text, font_name='Times New Roman', font_size=24, x=5, y=5, anchor_x='left',anchor_y='baseline', color=(255, 0, 255, 128)).draw()
        # render to screen
        self.win.flip()

    def capture_screen(self):
        glReadBuffer(GL_FRONT)
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, self.buffer)
        image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=self.buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        image = np.array(image)
        # convert RGB to BGR for OpenCV
        new_image = np.copy(image)
        new_image[:,:,0] = image[:,:,2]
        new_image[:,:,2] = image[:,:,0]
        return new_image

    def get_3D_render(self, shape, triangles):
        self.render(shape, triangles)
        return self.capture_screen()

    def exit(self):
        self.win.close()



class Visualizer(object):
    def __init__(self, draw_error=True):
        self.baseshapes = load_processed_baseshapes()
        self.triangles = load_triangles()
        self.renderer = Renderer()
        self.draw_error = draw_error

    def visualize(self, image, e_real, e_fake):
        shape_fake = calc_shape(self.baseshapes, e_fake)
        shape_real = calc_shape(self.baseshapes, e_real)

        self.renderer.render(shape_real, self.triangles)
        img_real = self.renderer.capture_screen()
        self.renderer.render(shape_fake, self.triangles)
        img_fake = self.renderer.capture_screen()

        # result
        new_img = np.zeros((300,900,3), dtype=np.uint8)
        if image is not None:
            new_img[:,0:300,:] = cv2.resize(image, (300,300), interpolation=cv2.INTER_CUBIC)
        new_img[:,300:600,:] = img_real[52:352,170:470,:]
        new_img[:,600:900,:] = img_fake[52:352,170:470,:]
    
        # error text
        if self.draw_error:
            error = np.sum(np.square(e_real-e_fake))
            txt = "error: {:.4f}".format(error)
            cv2.putText(new_img, txt, (10,280), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 0, 255), 1)
        
        return new_img

    def exit(self):
        self.renderer.exit()

    def restart(self):
        self.renderer = Renderer()


#--------------------------------------------------
def draw_error_bar_plot(e_real, e_fake, final_size=(900,100)):
    error = np.round(np.abs(e_real - e_fake) * 100.0).astype(np.int32)
    eg = np.round(e_real * 100.0).astype(np.int32)
    ef = np.round(e_fake * 100.0).astype(np.int32)
    # draw 46 bars
    img = np.zeros((460, 220, 3), dtype=np.uint8) + 255
    for i in range(46):
        # draw the error bars
        y1 = 2 + i*10
        y2 = y1 + 6
        x1 = 120
        x2 = 120 + error[i]
        img = cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), cv2.FILLED)
        img = cv2.putText(img, "{:d}".format(i+1), (105, y1+5), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
        # draw e_fake bars
        x1 = 0
        x2 = ef[i]
        img = cv2.rectangle(img, (x1,y1), (x2,y1+3), (0, 255, 0), cv2.FILLED)
        # draw e_real bars
        x2 = eg[i]
        img = cv2.rectangle(img, (x1,y1+3), (x2,y2), (0, 0, 255), cv2.FILLED)
    img = cv2.transpose(img)
    img = cv2.flip(img, 0)
    ret = cv2.resize(img, final_size)
    return ret
 

if __name__ == "__main__":
    triangles = load_triangles()
    baseshapes = load_processed_baseshapes()
    renderer = Renderer()
    e = np.zeros(46, dtype=np.float32)
    shape = calc_shape(baseshapes, e)
    renderer.render(shape, triangles)
    img = renderer.capture_screen()
    renderer.exit()
    cv2.imshow("image", img)
    cv2.waitKey()