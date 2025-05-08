import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import imageio 
import OpenEXR
import Imath
# import imageio.v2 as imageio

# def load_hdr_image(path):
#     # imageio returns an array with HDR values.
#     return np.array(imageio.imread(path)).astype(np.float32)


def load_image(path):
    return np.array(Image.open(path)) / 255.0

def save_image(image, path):
    Image.fromarray((image * 255).astype('uint8')).save(path)


'''Standard RGB is a non linear color space good for display. But the linear color space is needed for the lighting and shading purpose'''
def srgb_to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def linear_to_srgb(linear):
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * (linear ** (1/2.4)) - 0.055
    )

def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(norm > 0, v / norm, 0)  #The second condition is for keeping aways some error

def reflect(incident, normal):
    return incident - 2 * np.sum(incident * normal, axis=-1, keepdims=True) * normal


def load_hdri(hdri_path):
    print("Loading Environment Map...")
    exr_file = OpenEXR.InputFile(hdri_path)

    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = exr_file.channels(['R', 'G', 'B'], Imath.PixelType(Imath.PixelType.FLOAT))
    # print(type(channels))

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

thetaA = [2.75194496563142, 1.781368137461315, 2.499290487558739, 2.6374491611975723, 1.6863991048764237] 

phiA = [1.4929433742281337 ,2.562227343425438,


2.1979087835949316,

2.029780187036543,

 
3.838425068944307]
def extract_light(hdri, num_samples):
    print("Extracting Lighting...")
    height, width, _ =hdri.shape
    lights=[]

    for _ in range(num_samples):
        theta=np.random.uniform(0, np.pi)
        phi=np.random.uniform(0, 2*np.pi)

        # theta = thetaA[_]
        # phi = phiA[_]

        # print(theta, phi)

        x=np.sin(theta)*np.cos(phi)
        y=np.sin(theta)*np.sin(phi)
        z=np.cos(theta)

        direction=np.array([x,y,z])
        # print(direction)


        #pixels (u,v) in hdri map
        u=int((phi/(2*np.pi))*width)
        # print(u)
        v=int((theta/(2*np.pi))*height)

        color=hdri[v,u, :3]
        # print(color)
        intensity=np.linalg.norm(color)         #RMS

        lights.append({
            'type':'directional',
            'direction':normalize_hdri(direction),
            'color':color/intensity if intensity!=0 else np.zeros(3),
            'intensity':intensity
        })
    # print(lights)
    return lights

def multiply(albedo_path, normal_path, alpha_path):
    albedo_srgb = load_image(albedo_path)
    if albedo_srgb.shape[-1] == 4:
        albedo_srgb = albedo_srgb[..., :3]
    albedo_linear = srgb_to_linear(albedo_srgb)

    normal_png = load_image(normal_path)
    print(normal_png.shape)
    normal = 2.0 * normal_png - 1.0  
    normal = normalize(normal)

    # specular_map = load_image(specular_map_path)
    mask = load_image(alpha_path)

    height, width, _ = albedo_linear.shape
    result_1 = np.zeros((height, width, 3))
    result_2 = np.zeros((height, width, 3))
    result_3 = np.zeros((height, width, 3))

    for y in range(height):
        for x in range(width):

            albedo_pixel = albedo_linear[y, x]
            normal_pixel = normal[y, x]
            alpha_pixel = mask[y, x]

            result_1[y, x] = albedo_pixel * normal_pixel
            result_2[y, x] = alpha_pixel * albedo_pixel
            result_3[y, x] = alpha_pixel * normal_pixel

    
    result_1 = np.clip(result_1, 0, 1)
    result_2 = np.clip(result_2, 0, 1)
    result_3 = np.clip(result_3, 0, 1)
    # resul_srgb = linear_to_srgb(result)

    save_image(result_1, "im1.png")
    save_image(result_2, "im2.png")
    save_image(result_3, "im3.png")




def relight_with_specular_map(albedo_path, normal_path, specular_map_path, alpha_path, hdri_path, output_path):
    print("Starting...")
    albedo_srgb = load_image(albedo_path)
    if albedo_srgb.shape[-1] == 4:
        albedo_srgb = albedo_srgb[..., :3]
    albedo_linear = srgb_to_linear(albedo_srgb)

    normal_png = load_image(normal_path)
    print(normal_png.shape)
    normal = 2.0 * normal_png - 1.0  
    normal = normalize(normal)

    specular_map = load_image(specular_map_path)
    mask = load_image(alpha_path)

    # lights = [
    #     {
    #         'type': 'directional',
    #         'direction': normalize(np.array([-0.7, 0.7, 0.5])),  # top-left direction
    #         'color': np.array([1, 1, 1]),
    #         'intensity': 1
    #     },
    #     {
    #         'type': 'directional',
    #         'direction': normalize(np.array([0.7, 0.7, 0.5])),  # top-right direction
    #         'color': np.array([0, 1, 0]),
    #         'intensity': 1
    #     }
    # ]
    hdri=hdri=load_hdri(hdri_path)
    lights=extract_light(hdri, num_samples=5)
    print("Lights Extacted...")
    
    '''Ambient term. The ambient term models the soft, indirect lighting that hits all surfaces equally, regardless of their orientation. 
    It's used to avoid completely black areas in shadows and make objects look more realistic.'''
    ambient = np.array([0.15, 0.2, 0.25])    

    shininess = 10.0                #can change this

    height, width, _ = albedo_linear.shape
    result = np.zeros((height, width, 3))

    view_dir = np.array([0, 0, 1])


    print("Now Relighting...")
    for y in range(height):
        for x in range(width):

            # if mask[y, x] <= 0.6:   #Mask is the seg/alpha-matte mask.
            #     result[y, x] = np.zeros(3) 
            #     continue


            albedo_pixel = albedo_linear[y, x]
            normal_pixel = normal[y, x]
            specular_strength_pixel = specular_map[y, x] * 0.1

            diffuse = np.zeros(3)
            specular = np.zeros(3)

            for light in lights:
                l_dir = light['direction']
                ndotl = max(np.dot(normal_pixel, l_dir), 0.0)
                diffuse += ndotl * light['intensity'] * light['color']

                h = normalize(view_dir + l_dir)
                ndoth = max(np.dot(normal_pixel, h), 0.0)
                specular += (ndoth ** shininess) * specular_strength_pixel * light['color']

            ambient_term = ambient * albedo_pixel
            # result[y, x] = ((diffuse * albedo_pixel) + ambient_term + specular) 
            result[y, x] = ((diffuse * albedo_pixel) + ambient_term + specular) * mask[y, x]

    result = np.clip(result, 0, 1)
    result_srgb = linear_to_srgb(result)
    save_image(result_srgb, output_path)
    print("Done...")

# multiply(albedo_path="inputs/albedo.png", normal_path="inputs/normal.png", alpha_path="inputs/alpha.png")

relight_with_specular_map(
    albedo_path="inputs/albedo.png",
    normal_path="inputs/normal.png",
    specular_map_path="inputs/specular.png",
    alpha_path="inputs/alpha.png",
    hdri_path="/home/stud1/Aniket/PBR_relight/relighting-42/inputs/result.exr",
    output_path="output_phong2.png")
