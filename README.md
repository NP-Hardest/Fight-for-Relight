# Fight-for-Relight

Python implementation for the following work: https://drive.google.com/file/d/1LHcKlBnNC_ZUh2JpqSHXVHcmQZc4HHU3/view?usp=drivesdk, done as an undergraduate project under Prof. Tushar Sandhan, IIT Kanpur

Put all the maps (Albedo, Surface Normal, Specular, Roughness) in the inputs folder, if you want to relight a single image. To relight video frames, place the intrinsics for each frame in the maps folder. For example: The ith frame intrinsics should be placed in the  folder `/maps/result{i}`.

Run `python cook.py` or `python phong.py` to relight a single image (using Cook-Torrance or Blinn-Phong models respectively). 

Again, double check the input and output directories for relighting videos or images.



