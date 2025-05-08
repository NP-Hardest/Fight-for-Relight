import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import OpenEXR
import Imath
import math


def load_image(path):
    img = Image.open(path)
    new_size = (img.width // 2, img.height // 2)
    # if new_size != (32, 16):
        # img = img.resize(new_size, Image.Resampling.LANCZOS)
    return np.array(img) / 255.0

def save_image(image, path):
    Image.fromarray((image * 255).astype('uint8')).save(path)

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
    return np.where(norm > 0, v / norm, 0)


def load_hdri(hdri_path):
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
    im = Image.fromarray((rgb_image * 255).astype(np.uint8))
    im.save("mas.png")
    return rgb_image

def normalize_hdri(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def reflect(incident, normal):
    return incident - 2 * np.sum(incident * normal, axis=-1, keepdims=True) * normal

dir = [[-2, 1, 1], [2, 1, 1]]

TP = [(0.7642582478336393, 4.220521622608128), (1.2124105560742837, 0.13338020487506458)]





def extract_light(hdri, num_samples):
    height, width, _ =hdri.shape
    # print(hdri.shape)
    lights=[]

    for _ in range(num_samples):
        
        # theta=np.random.uniform(0, np.pi)
        # phi=np.random.uniform(0, 2*np.pi)


        theta, phi = TP[_]

        print(f"{theta}, {phi}")


        x=np.sin(theta)*np.cos(phi)
        y=np.sin(theta)*np.sin(phi)
        z=np.abs(np.cos(theta))                 # to avoid negative z



        direction=np.array([x,y,z])


        #pixels (u,v) in hdri map
        u=int((phi/(2*np.pi))*width)

        v=int((theta/(2*np.pi))*height)

        color=hdri[v, u, :3]

        intensity=np.linalg.norm(color)         #RMS
        color=color/intensity if intensity!=0 else np.zeros(3)


        lights.append({
            'type':'directional',
            'direction': normalize(direction),
            'color':color,
            'intensity':intensity
        })
    return lights



def relight(original_image, albedo_path, normal_path, specular_map_path, output_path, roughness_map_path, alpha_path, hdri_path):
    original = load_image(original_image)
    albedo_srgb = load_image(albedo_path)
    if albedo_srgb.shape[-1] == 4:
        albedo_srgb = albedo_srgb[..., :3]
    albedo_linear = srgb_to_linear(albedo_srgb)

    normal_png = load_image(normal_path)
    normal = 2.0 * normal_png - 1.0  
    normal = normalize(normal)

    specular_map = load_image(specular_map_path)
    roughness_map = load_image(roughness_map_path)
    mask = load_image(alpha_path)

    hdri = load_image(hdri_path)
    # hdri = hdri/255
    # hdri=load_hdri(hdri_path)
    # print(np.max(hdri))
    print("Extracting Lighting...")
    lights=extract_light(hdri, num_samples=2)
    print("Lights Ready")
    # print(lights)




    if specular_map.ndim == 3:
        specular_map = specular_map[..., 0]


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

    
    # lights = [
    #     {'type': 'directional', 'direction': normalize(np.array([-0.26169773, -0.02832835, 0.16473406])), 'color': np.array([0, 0, 1]), 'intensity': 1},
          
    #  {'type': 'directional', 'direction': normalize(np.array([-0.36880833,  0.61242641, 0.19922408])), 'color': np.array([1, 0, 1]), 'intensity': 1}
    #  ]
    

    ambient = np.array([0.15, 0.2, 0.25])           # Ambient term.

    height, width, _ = albedo_linear.shape
    result = np.zeros((height, width, 3))
    result2 = np.zeros((height, width, 3))

    view_dir = np.array([0, 0, 1])

    def xi(x):
        if x > 0: 
            return 1
        else:
            return 0

    print("Relighting Now...")

    for y in range(height):
        for x in range(width):

            result2[y, x] = normal[y, x] * mask[y, x]
            # if mask[y, x] <= 0.99:
            #     result[y, x] = np.zeros(3)
            #     continue

            albedo_pixel = albedo_linear[y, x]
            normal_pixel = normal[y, x]


            F0 = specular_map[y, x] * 0.1
            roughness = roughness_map[y, x]
            alpha = roughness * roughness  

            diffuse = np.zeros(3)
            specular = np.zeros(3)

            for light in lights:
                l_dir = light['direction']
                ndotl = max(np.dot(normal_pixel, l_dir), 0.0)


                # ndotl = max(np.dot(normal_pixel, l_dir), 0.0)
                ndotv = max(np.dot(normal_pixel, view_dir), 0.0)

                h = normalize(view_dir + l_dir)
                ndoth = np.dot(normal_pixel, h)
                vdoth = np.dot(view_dir, h)
                            
                denom = ndoth * ndoth * (alpha * alpha - 1) + 1             # GGX  normal distribution function d.
                D = (xi(ndoth) * alpha * alpha) / (np.pi * denom * denom)

                F = F0 + (1 - F0) * ((1 - vdoth) ** 5)                     # Fresnel term using Schlick's approximation.

                
                k = (roughness + 1) ** 2 / 8.0          # Geometry term using Smith's method with Schlick-GGX.
                G_V = ndotv / (ndotv * (1 - k) + k) if ndotv > 0 else 0.0
                G_L = ndotl / (ndotl * (1 - k) + k) if ndotl > 0 else 0.0
                G = G_V * G_L

                # diffuse += ndotl * light['intensity'] * light['color']
                diffuse += ndotl * (1 - F) * light['intensity'] * light['color']
                spec = (D * F * G) / (4 * ndotv * ndotl + 0.001)
                specular += spec * light['color'] * light['intensity']

            ambient_term = ambient * albedo_pixel

            r1 = ((diffuse * albedo_pixel) + ambient_term + specular) * mask[y, x]
            r2 = (original[y, x][:3]) * (1 - mask[y, x])

            r1 = np.where(
                r1 <= 0.0031308,
                r1 * 12.92,
                1.055 * (r1 ** (1/2.4)) - 0.055
            )
            # result[y, x] = r1
            result[y, x] = r1 
            # result[y, x] = ((diffuse * albedo_pixel) + ambient_term + specular) * mask[y, x]
            # result[y, x] = (diffuse * albedo_pixel) + ambient_term + specular

    
    result = np.clip(result, 0, 1)
    save_image(result, output_path)
    # result_srgb = linear_to_srgb(result)
    # save_image(result_srgb, output_path)           
    # save_image(result2, "output_bg.png")      
    # 

for i in range(1, 51):
    print(f"Relighting frame {i}...")
    relight(
        # original_image="inputs/in.png",
        # albedo_path="inputs/albedo.png",
        # normal_path="inputs/normal.png",
        # specular_map_path="inputs/specular.png",
        # roughness_map_path="inputs/roughness.png",
        # alpha_path="inputs/alpha.png",
        # output_path="outputs/output_cook_1.png",
        # hdri_path="inputs/map.png"
        original_image=f"maps/result{i}/{i}.png",
        albedo_path=f"maps/result{i}/albedo.png",
        normal_path=f"maps/result{i}/normal.png",
        specular_map_path=f"maps/result{i}/specular.png",
        roughness_map_path=f"maps/result{i}/roughness.png",
        alpha_path=f"maps/result{i}/alpha.png",
        output_path=f"outputs/vid/{i}.png",
        hdri_path="inputs/map2.png"
        # hdri_path="inputs/result.exr"
    )
