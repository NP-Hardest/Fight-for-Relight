import numpy as np
import imageio
import imageio.v3 as iio
import OpenEXR
import Imath




def load_hdri(hdri_path):
    # hdri = iio.imread(hdri_path, format="EXR") 
    # # cv2.imread(hdri_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)  
    # img = cv2.imread(hdri_path, cv2.IMREAD_ANYCOLOR)
    exr_file = OpenEXR.InputFile(hdri_path)

    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

# Read channels (e.g., 'R', 'G', 'B')
    channels = exr_file.channels(['R', 'G', 'B'], Imath.PixelType(Imath.PixelType.FLOAT))
    print(type(channels))

    # Convert to numpy array
    red = np.frombuffer(channels[0], dtype=np.float32).reshape(height, width)
    # print(red)
    green = np.frombuffer(channels[1], dtype=np.float32).reshape(height, width)
    blue = np.frombuffer(channels[2], dtype=np.float32).reshape(height, width)

    rgb_image = np.stack((red, green, blue), axis=-1)

    # print("EXR shape:", rgb_image.shape)  # (height, width, 3)
    return rgb_image

def normalize_hdri(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def extrat_ligth(hdri, num_samples):
    height, width, _ =hdri.shape
    lights=[]

    for _ in range(num_samples):
        theta=np.random.uniform(0, np.pi)
        phi=np.random.uniform(0, 2*np.pi)

        x=np.sin(theta)*np.cos(phi)
        y=np.sin(theta)*np.sin(phi)
        z=np.cos(theta)

        direction=np.array([x,y,z])
        print(direction)


        #pixels (u,v) in hdri map
        u=int((phi/(2*np.pi))*width)
        # print(u)
        v=int((theta/(2*np.pi))*height)

        color=hdri[v,u, :3]
        print(color)
        intensity=np.linalg.norm(color)

        lights.append({
            'type':'directional',
            'direction':normalize_hdri(direction),
            'color':color/intensity if intensity!=0 else np.zeros(3),
            'intensity':intensity
        })
    
    print(lights)

    return lights

hdri_path='/home/stud1/Aniket/PBR_relight/relighting-42/inputs/54_ev-25.exr'

# if os.path.exists(hdri_path):
#     print("check permission")
#     if os.access(hdri_path, os.R_OK):
#         print("permisson ok")
#     else:
#         print("path not accesable")

# else:
#     print("file not found")

hdri=load_hdri(hdri_path)
# print(hdri.shape)

extrat_ligth(hdri, num_samples=2)