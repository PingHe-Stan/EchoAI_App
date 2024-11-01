from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
import json
import random
from PIL import Image
import base64
import io
import cv2
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Elo rating constants
K = 32

# Starting Elo ratings for each generation method
elo_gen1 = 1500
elo_gen2 = 1500

# Paths
TRAIN_RESIZED_PATH = './train_resized'
IMAGES_PATH = os.path.join(TRAIN_RESIZED_PATH, 'images')
MASKS_PATH = os.path.join(TRAIN_RESIZED_PATH, 'masks')
FCBFormer_PATH = "./train_resized/FCBFormer_8/"

# List of image pairs for tracing selector
gt_list = sorted([f"{root.rstrip('/')}/{f}" for root,_,files in os.walk(FCBFormer_PATH) for f in files if "_gt" in f])
pred_list = sorted([f"{root.rstrip('/')}/{f}" for root,_,files in os.walk(FCBFormer_PATH) for f in files if "_pred" in f])
raw_list = sorted([f"{root.rstrip('/')}/{f}" for root,_,files in os.walk(FCBFormer_PATH) for f in files if "_raw" in f])

image_pairs = [(i,j,k) for i, j, k in zip(gt_list,pred_list,raw_list) if i.split("_")[-2] == j.split("_")[-2] == k.split("_")[-2]]

# Global variables
current_pair_index = 0
user_selections = {}
selected_app = None

# Secret key
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')

# Dummy user store
users = {
    'sk': generate_password_hash('echo'),
    'lauren': generate_password_hash('echo'),
    'stan': generate_password_hash('echo'),
    'wei': generate_password_hash('echo'),
    'faith': generate_password_hash('echo'),
    'mariella': generate_password_hash('echo'),
    'rakhika': generate_password_hash('echo'),
    'luc': generate_password_hash('echo'),
    'mark': generate_password_hash('echo')
}


def get_progress_file():
    """Generate the progress file path based on the logged-in user."""
    username = session.get('username')
    return os.path.join(TRAIN_RESIZED_PATH, f'{username}_progress.json')

def get_selection_file():
    """Generate the progress file path based on the logged-in user."""
    username = session.get('username')
    return os.path.join(TRAIN_RESIZED_PATH, f'{username}_selection.json')

def get_next_image_pair():
    """Get the next unscored image-mask pair and the progress."""
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {}
    
    image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')])
    total_images = len(image_files)
    completed_images = len(progress)
    
    
    for image_file in image_files:
        if image_file not in progress:
            image_path = os.path.join(IMAGES_PATH, image_file)
            mask_path = os.path.join(MASKS_PATH, image_file)
            if os.path.exists(image_path) and os.path.exists(mask_path):
                return (image_path, mask_path), completed_images, total_images
    
    return None, completed_images, total_images

def update_progress(image_path, mask_path, score):
    """Update the progress file with the scored image."""
    progress_file = get_progress_file()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {}

    image_file = os.path.basename(image_path)
    progress[image_file] = {
        'image_path': image_path,
        'mask_path': mask_path,
        'score': score
    }

    with open(progress_file, 'w') as f:
        json.dump(progress, f)

def image_to_base64(image_path):
    """Convert an image file to a base64 string."""
    with Image.open(image_path) as img:
        max_size = (800, 800)
        img.thumbnail(max_size)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def overlay_mask(image_path, mask_path, opacity=0.35):
    """Overlay the mask on the image with given opacity."""
    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        if mask.mode != 'RGBA':
            mask = mask.convert('RGBA')
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay.paste(mask, (0, 0), mask.split()[3])
        overlay = Image.blend(Image.new('RGBA', img.size), overlay, opacity)
        result = Image.alpha_composite(img.convert('RGBA'), overlay)
        result = result.convert('RGB')
        result = result.resize((450,450))
        buffer = io.BytesIO()
        result.save(buffer, format='JPEG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def calculate_elo(winner_elo, loser_elo):
    expected_score_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner_elo = winner_elo + K * (1 - expected_score_winner)
    new_loser_elo = loser_elo + K * (0 - (1 - expected_score_winner))
    return new_winner_elo, new_loser_elo
    
def draw_contours(mask_path, background_path, color=(255, 0, 0), thickness=2):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 122).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if isinstance(background_path, str):
        background = cv2.imread(background_path)
    else:
        background = np.array(background_path)
    cv2.drawContours(background, contours, -1, color, thickness)
    result = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    return result, contours

def load_images(image_pair):
    contour_image1, contours1 = draw_contours(image_pair[0], image_pair[2], color=(0, 0, 255), thickness=1)
    contour_image2, contours2 = draw_contours(image_pair[1], image_pair[2], color=(0, 0, 255), thickness=1)
    background = np.array(contour_image1)
    cv2.drawContours(background, contours2, -1, (0, 0, 255), 1)
    contour_image_mixed = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    contour_image1 = contour_image1.resize((600,600), Image.Resampling.LANCZOS)
    contour_image2 = contour_image2.resize((600,600), Image.Resampling.LANCZOS)
    contour_image_mixed = contour_image_mixed.resize((800,800), Image.Resampling.LANCZOS)
    return contour_image1, contour_image2, contour_image_mixed

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Render login page and handle authentication."""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_password = users.get(username)
        
        # Update image_pairs global, reset user_selections & current_pair_index
        global gt_list,pred_list,raw_list, image_pairs, elo_gen1, elo_gen2, user_selections, current_pair_index
        user_selections = {} # Clean up previous selection
        current_pair_index = 0
        elo_gen1 = 1500
        elo_gen2 = 1500
        
        if user_password and check_password_hash(user_password, password):
            session['username'] = username
            image_pairs = [(i,j,k) for i, j, k in zip(gt_list,pred_list,raw_list) if i.split("_")[-2] == j.split("_")[-2] == k.split("_")[-2]]
            original_len = len(image_pairs)
            
            # Update global variable if there is history file
            if os.path.exists(get_selection_file()):
                with open(get_selection_file(), 'r') as file:
                    user_selections = json.load(file)
                image_pairs = [item for item in image_pairs if os.path.basename(item[0]).split("_")[0] not in user_selections.keys()]
                # Restore the elo gen score from the last comparison
                last_key = list(user_selections.keys())[-1]
                elo_gen1 = user_selections[last_key]["elo_ratings"][0]
                elo_gen2 = user_selections[last_key]["elo_ratings"][1]
            
            print(f"{username} has completed {original_len - len(image_pairs)} out of {original_len}")
            print(f"The resumed elo scoring is {elo_gen1} and {elo_gen2} respectively")
            
            return redirect(url_for('app_selector'))
        else:
            return 'Invalid credentials', 401
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout the user."""
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/app_selector', methods=['GET'])
def app_selector():
    """Render app selector page."""
    if 'username' not in session:
        return 'Credentials required. Please <a href="/login">login</a>.', 403
    
    return render_template('app_selector.html')

@app.route('/segmentation_rater', methods=['GET', 'POST'])
def segmentation_rater():
    """Render the Segmentation Rater app."""
    if 'username' not in session:
        return 'Credentials required. Please <a href="/login">login</a>.', 403
    
    if request.method == 'GET':
        return render_template('segmentation_rater.html')
    
    elif request.method == 'POST':
        try:
            action = request.json.get('action')
            if action == 'get_image_pair':
                image_pair, completed_images, total_images = get_next_image_pair()
                if image_pair:
                    image_path, mask_path = image_pair
                    try:
                        image_base64 = image_to_base64(image_path)
                        overlay_base64 = overlay_mask(image_path, mask_path)
                        return jsonify({
                            'image': image_base64,
                            'overlay': overlay_base64,
                            'image_path': image_path,
                            'mask_path': mask_path,
                            'completed': completed_images,
                            'total': total_images
                        })
                    except Exception as e:
                        return jsonify({'error': str(e)}), 500
                else:
                    return jsonify({'message': "Thank you so much! All evaluation have been completed!!!"})
            elif action == 'score':
                update_progress(request.json['image_path'], request.json['mask_path'], request.json['score'])
                return jsonify({'success': True})
            else:
                return jsonify({'error': 'Invalid action'}), 400
        except Exception as e:
            return jsonify({'error': 'Server error occurred'}), 500

@app.route('/tracing_selector', methods=['GET'])
def tracing_selector():
    """Render the Tracing Selector app."""
    if 'username' not in session:
        return 'Credentials required. Please <a href="/login">login</a>.', 403
    
    global current_pair_index, image_pairs
    
    if current_pair_index < len(image_pairs):
        img1, img2, img_mix = load_images(image_pairs[current_pair_index])
        img1_name = os.path.basename(image_pairs[current_pair_index][0])
        img2_name = os.path.basename(image_pairs[current_pair_index][1])
        img_mix_name = os.path.basename(image_pairs[current_pair_index][0]).split("_")[0] + '_mixed.jpg'
        os.makedirs("static", exist_ok=True)
        img1.save(os.path.join('static', img1_name))
        img2.save(os.path.join('static', img2_name))
        img_mix.save(os.path.join('static', img_mix_name))
        progress = int((current_pair_index / len(image_pairs)) * 100)

        buttons = [ 
                {'display': 'Select 👇 Tracing', 'img': img1_name, 'label':'img1'},
                {'display': 'Select 👇 Tracing', 'img': img2_name, 'label':'img2'}
                ]

        random.shuffle(buttons)

        return render_template('tracing_selector.html', progress=progress, buttons = buttons, img1_name=img1_name, img2_name=img2_name, img3_name=img_mix_name)
    else:
        return redirect(url_for('results'))

@app.route('/select_image', methods=['POST'])
def select_image():
    """Handle the selection of images in the Tracing Selector app."""
    global current_pair_index, elo_gen1, elo_gen2

    selection = request.json['selection']
    selected_image = image_pairs[current_pair_index][0] if selection == 'img1' else image_pairs[current_pair_index][1]
    image_key = os.path.basename(selected_image).split("_")[0]  # Extract the first few digits as the key
    comment = "NA"
    if selection == 'img1':
        elo_gen1, elo_gen2 = calculate_elo(elo_gen1, elo_gen2)
    elif selection == 'img2':
        elo_gen2, elo_gen1 = calculate_elo(elo_gen2, elo_gen1)
    elif selection == 'failed':
        comment = 'failed'
    elif selection == 'acceptable':
        comment = 'acceptable'
    else:
        comment = 'invalid'

    current_pair_index += 1

    # Store the selection using the new format (dictionary key is the first few digits)
    user_selections[image_key] = {
        'selection': selected_image,
        'elo_ratings': (elo_gen1, elo_gen2),
        'comment': comment
    }

    comment = "NA"

    # Save the output data to a JSON file
    with open(get_selection_file(), 'w') as f:
        json.dump(user_selections, f, indent=4)

    return jsonify({'next': url_for('tracing_selector')})


@app.route('/update_opacity', methods=['POST'])
def update_opacity():
    try:
        data = request.json
        image_path = data.get('image_path')
        mask_path = data.get('mask_path')
        opacity = data.get('opacity')
        
        if image_path and mask_path:
            overlay_base64 = overlay_mask(image_path, mask_path, opacity)
            return jsonify({'overlay': overlay_base64})
        else:
            return jsonify({'error': 'Invalid image or mask path'}), 400
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': 'Server error occurred'}), 500

@app.route('/results')
def results():
    """Render the results of the Tracing Selector."""
    return f"Elo Rating - Generation 1: {elo_gen1}<br>Elo Rating - Generation 2: {elo_gen2}"

if __name__ == '__main__':
    app.run(debug=False, port=80)
