# Pa Omninet: A Retraining-Free, Generalizable Deep Learning Framework for Robust Photoacoustic Image Reconstruction


This code was used as part of the work presented in 

Olivier J.M. Stam, Kalloor Joseph Francis* and Navchetan Awasthi* "PA OmniNet: A Retraining-Free, Generalizable Deep Learning
Framework for Robust Photoacoustic Image Reconstruction” Available at Photoacoustics: https://www.sciencedirect.com/science/article/pii/S2213597925000631


* Authors contributed equally

[//]: # (**The raw measurement data for the experimental experiments is not provided and can be requested.)
**Please contact if you find any mistakes or if you need any help regarding the codes.

#Jupyter notebook implementation for creating figures for the OADAT dataset: Figures_OADAT.ipynb\
#Jupyter notebook implementation for reformatting mice data: Altermousedata.ipynb\
#Jupyter notebook implementation for evaluating the mice sub-datasets: evaluation_mice_paper.ipynb\
#Jupyter notebook implementation for creating figures for the mice sub-datasets: Figures_Mice.ipynb

App: \
#Python implementation for the model used in the backend of the app: App_model.py\
#Python implementation for the backend of the app: PA_OmniNet_backend.py\
#Python implementation for the app: PA_OmniNet_app.py

Helper functions:\
#Python implementation for helper functions around all files: helper_functions.py\
#Python implementation for testing functions in evaluation files: testing_functions.py

Mice:\
#Python implementation for training the U-nets on the Mice data: train_Unet_ouse.py\
#Python implementation for training the PA OmniNet on the Mice data: train_PAOmniNet_Mouse.py
OADAT:\
#Python implementation for training PA OmniNet on SWFD Semi, SWFD Multi and MSFD: train_PAOmniNet_OADAT.py\
#Python implementation for training PA OmniNet on SCD: train_PAOmniNet_OADATSCD.py\
#Python implementation for training U-net on OADAT SWFD Semi, SWFD Multi and MSFD: train_Unet_OADAT.py\
#Python implementation for training U-net on OADAT SCD: train_Unet_OADATSCD.py\
#Python implementation for retrieving numerical results for the OADAT SWFD Semi, SWFD Multi and MSFD datasets on the U-net: test_Unet_OADAT.py\
#Python implementation for retrieving all the images of SWFD Semi, SWFD Multi and MSFD: PA_OmniNet_all_images.py\
#Python implementation for retrieving all the images of SCD: PA_OmniNet_all_images_SCD.py\
#Python implementation for retrieving numerical results for the OADAT datasets on the PA OmniNet for SWFD Semi, SWFD Multi and MSFD: test_PAOmniNet_OADAT.py\
#Python implementation for retrieving numerical results for the OADAT datasets on the PA OmniNet for SCD: test_PAOmniNet_OADATSCD.py\
#Python implementation for  retrieving all the images of the OADAT sub-datasets on the U-net: Unet_all_images.py


Backend models: \
#Python implementation for PA OmniNet testing model: testing_model.py\
#Python implementation for U-net backbone: unet1.py


#Environment install: environment.yml




