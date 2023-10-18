#%% 
from skimage.transform import hough_line, hough_line_peaks
from pylab import rcParams
import numpy as np
import cv2
import skimage.io
import math
from skimage.draw import circle_perimeter
import matplotlib.pyplot as plt
import copy

# Get the interface by erosion and dilatation 

def erode_and_dilate(input):   
    ''' Allow to g et the edges of the object '''    
    kernel = np.ones((3,3), np.uint8)  
    eroded = input - cv2.erode(input, kernel=kernel)
    dilated = input - cv2.dilate(input, kernel, iterations=1)  
    return dilated - eroded 

def get_interface(wire, droplet, erode=True):
    line = cv2.bitwise_and(droplet, droplet, mask=wire)
    line = cv2.dilate(line, kernel=np.ones((5,5), np.uint8), iterations=2)
    if erode:
        line = cv2.erode(line, kernel=np.ones((3,3), np.uint8), iterations=2)
    return line

def plot_interface(image, droplet_edge, wire_edge, interface):
    f, axarr = plt.subplots(1,4)
    rcParams['figure.figsize'] = 30,30
    axarr[0].imshow(image, cmap='Greys_r')
    axarr[1].imshow(droplet_edge, cmap='Greys_r')
    axarr[2].imshow(wire_edge, cmap='Greys_r')
    axarr[3].imshow(interface, cmap='Greys_r')



# Get points and line

def detect_lines(image, tested_angles=180, threshold=0.3, display=False):
    hspace, theta, dist = hough_line(image, np.linspace(-np.pi / 2, np.pi / 2, tested_angles))
    # Find peaks in the Hough space
    hspace_peaks, angles, distances = hough_line_peaks(hspace, theta, dist, threshold=threshold)

    detected_lines = []
    for angle, dist in zip(angles, distances):
        y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)
        detected_lines.append((y0, y1))

    y0, y1 = detected_lines[0]
    a = (y1-y0)/image.shape[1]
    b = y0
    # coordonnates within the image
    x0, x1 = -b/a, (image.shape[1]-b)/a

    if display:
        plt.figure()
        plt.imshow(image, cmap='gray')
        plt.plot((x0, x1), (0, image.shape[1]), 'r')
        plt.show()

    # print("equation of the line : y = " + str('{:.2f}'.format(a)) + "x + " + str('{:.2f}'.format(b)))

    return a, b, x0, x1



# Get the three points on the interface

def get_points(image_line, x0, x1):
    """ return the corrdonates of the extrem points and center point of the interface """
    profile = skimage.measure.profile_line(image_line, (0, x0), (image_line.shape[1], x1),linewidth=2)
    y_all_coor= np.linspace(0, image_line.shape[1], num=len(profile)) 
    x_all_coor= np.linspace(x0, x1, num=len(profile)) 

    profile_inv=profile[::-1]
    points=[]
    for i in range (profile.size):
        if profile[i] >0: #>0
            points.append(i)
            break
    for j in range (profile.size):        
        if profile_inv[j] >0:
            q=(profile.size)-j-1
            points.append(q)
            break

    y1_edge=y_all_coor[points[0]]
    y2_edge=y_all_coor[points[1]]
    x1_edge=x_all_coor[points[0]]
    x2_edge=x_all_coor[points[1]]

    x_center=(x2_edge+x1_edge)/2
    y_center=(y2_edge+y1_edge)/2

    # plt.figure()
    # plt.imshow(image_line)
    # plt.plot((x0, x1), (0, image_line.shape[1]), 'r')
    # plt.scatter((x1_edge, x2_edge, x_center), (y1_edge, y2_edge, y_center))
    # plt.show()

    return (x1_edge,y1_edge),(x2_edge,y2_edge),(x_center,y_center)


# Get the equation of the perpendicular to the interface 

def get_perpendicular(a, b, x_center, y_center, lenght):
    ''' a, b : coefficients of the equation of the line whose perpendicular you are looking for
        x_center, y_center : coordonates of the point where the perpendicular should cross the line
        lenght : lenght of the image'''
    a_perp = -1/a
    b_perp = y_center - a_perp*x_center
    x_perp = np.linspace(0, lenght, 2)
    y_perp = a_perp*x_perp + b_perp
    
    # print("equation of the line : y = " + str('{:.2f}'.format(a_perp)) + "x + " + str('{:.2f}'.format(b_perp)))

    return a_perp, b_perp



# Get the extrem points of the droplet

def get_droplet_points(a, b, edge_image):
    """ a, b : coefficients of the perpendicular line """
    # get the profil
    x1_perp = 0
    y1_perp = b
    x2_perp = edge_image.shape[1]
    y2_perp = a*x2_perp + b
    coor1_perp, coor2_perp = (y1_perp, x1_perp), (y2_perp, x2_perp)
    profile_perp = skimage.measure.profile_line(edge_image, coor1_perp, coor2_perp, linewidth=2)
    
    y_perp_all_coor = np.linspace(y1_perp, y2_perp, num=len(profile_perp)) 
    x_perp_all_coor = np.linspace(x1_perp, x2_perp, num=len(profile_perp)) 

    profile_perp_inv = profile_perp[::-1]
    profile_perp_drop_end = []

    for i in range (profile_perp.size):
        if profile_perp[i] >0: #>0
            profile_perp_drop_end.append(i)
            break

    for j in range (profile_perp.size):        
        if profile_perp_inv[j] > 0:
            q = profile_perp.size - j - 1
            profile_perp_drop_end.append(q)
            break

    x1_perp_drop_start = x_perp_all_coor[profile_perp_drop_end[0]]
    y1_perp_drop_start = y_perp_all_coor[profile_perp_drop_end[0]]
    x2_perp_drop_end = x_perp_all_coor[profile_perp_drop_end[1]]
    y2_perp_drop_end = y_perp_all_coor[profile_perp_drop_end[1]]

    # plt.figure()
    # plt.imshow(edge_image)
    # plt.scatter((x1_perp_drop_start, x2_perp_drop_end), (y1_perp_drop_start, y2_perp_drop_end), s=1000)
    # plt.show()

    return (x1_perp_drop_start, y1_perp_drop_start), (x2_perp_drop_end, y2_perp_drop_end), (x1_perp, y1_perp), (x2_perp, y2_perp)


# Get values 

def get_volume(h, a):
    return (1/6) * np.pi * h * (3*a**2 + h**2)

def get_area(h, a):
    return np.pi * (h**2 + a**2)

def error_contact_angle(contact_angle):
    if contact_angle is None:
        return None
    elif contact_angle > 2.35 or contact_angle < 1.4:
        return None
    else:
        return contact_angle

def get_contact_angle_v1(h, a):
    try:
        return error_contact_angle(np.pi - 2* np.arctan(a/h))
    except ValueError:
        return None
    except TypeError:
        return None

def get_contact_angle_v2(R, h):
    try:
        return error_contact_angle(np.pi/2 + math.asin(abs(R-h)/R))
    except ValueError:
        return None 
    except TypeError:
        return None

def get_contact_angle_v3(a,R):
    try:
        return error_contact_angle(np.pi/2 + math.acos(a/R))
    except ValueError:
        return None 
    except TypeError:
        return None
    


#%% Tests: get all the necesary measurements


# model_folder = "C:/Users/lucas.degeorge/Documents/trained_models"
# image_folder = "D:/Images_segmentation/Images/unlabeled_data/unlabeled_images"
# # image_folder = "D:/Images_nanomax/Contact_angle/by_frame/Video26"

# model_name = model_folder + "/unet_20230812_230324/unet_20230812_230324_best.pth"
# model = UNet(arguments).to(device)
# with open(model_name, 'rb') as f:
#     model.load_state_dict(torch.load(io.BytesIO(f.read())), strict=False)

# # image = image_folder + "/Video26_frame_00797.png"
# files = os.listdir(image_folder)
# image =  image_folder + "/" + random.choice(files)

# pred = predict(model, image, display=True, need_conversion=True).cpu().numpy()
# image = Image.open(image).convert("L")
# image = image.resize((512,512), resample=PIL.Image.NEAREST)
# image = np.asarray(image)

# pred_back = np.zeros_like(pred).astype('uint8')
# pred_droplet = np.zeros_like(pred).astype('uint8')
# pred_wire = np.zeros_like(pred).astype('uint8')

# pred_back[pred == 0] = 1
# pred_wire[pred == 1] = 1
# pred_droplet[pred == 2] = 1

# droplet_edge = erode_and_dilate(pred_droplet)
# wire_edge = erode_and_dilate(pred_wire)
# interface = get_interface(wire_edge, droplet_edge)
# plot_interface(pred, droplet_edge, wire_edge, interface)

# a, b, x0, x1 = detect_lines(interface, display=True)
# a_dl, b_dl, x0_dl, x1_dl = detect_lines(droplet_edge, display=True) 


# (x1_edge,y1_edge),(x2_edge,y2_edge),(x_center,y_center) = get_points(interface, x0, x1)
# (x1_edge_dl,y1_edge_dl),(x2_edge_dl,y2_edge_dl),(x_center_dl,y_center_dl) = get_points(droplet_edge, x0_dl, x1_dl)

# print((x1_edge,y1_edge),(x2_edge,y2_edge),(x_center,y_center))

# a_perp, b_perp = get_perpendicular(a, b ,x_center, y_center, interface.shape[1])

# (x1_perp_drop_start, y1_perp_drop_start), (x2_perp_drop_end, y2_perp_drop_end), (x1_perp, y1_perp), (x2_perp, y2_perp) = get_droplet_points(a_perp, b_perp, droplet_edge)


#%% Circle tests: 

# circle_info = cv2.HoughCircles(droplet_edge, cv2.HOUGH_GRADIENT,2,500, param1=300,param2=0.9,minRadius=0,maxRadius=0)[0,0].astype(np.int16)
# print(circle_info)

# R = circle_info[2]
# rr, cc = circle_perimeter(circle_info[1], circle_info[0], circle_info[2])

# droplet_with_circle = copy.deepcopy(droplet_edge)
# droplet_with_circle[rr, cc] = 128

# plt.figure()
# plt.imshow(droplet_with_circle)
# plt.show()

# droplet_with_circle = cv2.circle(droplet_edge, (circle_info[0], circle_info[1]), circle_info[2], (255, 0, 0), 2)


#%% Concact angle etc. 

# a = np.sqrt((y1_edge - y2_edge)**2 + (x1_edge - x2_edge)**2) / 2
# h = np.sqrt((y1_perp_drop_start - y2_perp_drop_end)**2 + (x1_perp_drop_start - x2_perp_drop_end)**2)

# print(a)
# # print(h)
# print(R-h)
# print(R)

# print("----------Contact angle----------")
# print("v1: " + str('{:.2f}'.format(get_contact_angle_v1(h, a))) + " rad = " + str('{:.2f}'.format(180*get_contact_angle_v1(h, a)/np.pi)) + "°")
# print("v2: " + str('{:.2f}'.format(get_contact_angle_v2(R, h))) + " rad = " + str('{:.2f}'.format(180*get_contact_angle_v2(R, h)/np.pi)) + "°")
# print("v3: " + str('{:.2f}'.format(get_contact_angle_v3(a,R))) + " rad = " + str('{:.2f}'.format(180*get_contact_angle_v3(a,R)/np.pi)) + "°")


# contact_angle = get_contact_angle(h, a)
# volume = get_volume(h, a)
# area = get_area(h, a)

# print("----------Measurements----------")
# print("Volume: " + str('{:.2f}'.format(volume)) + " px**3")
# print("Area: " + str('{:.2f}'.format(area)) + " px**2")
# print("Contact angle: " + str('{:.2f}'.format(contact_angle)) + " rad = " + str('{:.2f}'.format(180*contact_angle/np.pi)) + "°")


#%% Plots

# plt.figure()
# plt.xlim(0, image.shape[0])
# plt.ylim((image.shape[1], 0))
# plt.imshow(image, cmap='Greys_r')
# plt.plot((x0, x1), (0, interface.shape[1]), '-b')
# plt.scatter((x1_edge, x2_edge, x_center), (y1_edge, y2_edge, y_center))
# plt.scatter((x1_perp_drop_start, x2_perp_drop_end), (y1_perp_drop_start, y2_perp_drop_end))
# plt.plot((x1_perp_drop_start, x2_perp_drop_end), (y1_perp_drop_start, y2_perp_drop_end), '-r')
# # plt.plot(x1_edge, y1_edge, linewidth=3)
# # plt.plot(x2_edge, y2_edge, linewidth=3)
# plt.show()

