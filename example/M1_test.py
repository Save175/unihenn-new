import os
import sys
import gradio as gr
import torch
import numpy as np
from torchvision import datasets
from uni_henn import *
from models.model_structures import M1
from seal import *
from PIL import Image, ImageDraw, ImageFont

# ========== Setup ==========
root_dir = os.path.dirname(os.path.abspath(__file__))
context = sys.argv[1]  # Passed context

# Global Objects
m1_model = None
HE_m1 = None
MNIST_Img = Cuboid(1, 28, 28)

loaded_data = None
loaded_labels = None
encrypted_batch = None
decrypted_batch = None
num_of_data = 0

# ========== Functions ==========

def initialize_model_and_keys():
    global m1_model, HE_m1, num_of_data

    try:
        m1_model = M1()
        m1_model = torch.load(
            os.path.join(root_dir, "models/M1_model.pth"),
            map_location=torch.device('cpu'),
            weights_only=False
        )
        HE_m1 = HE_CNN(m1_model, MNIST_Img, context)
        num_of_data = int(context.number_of_slots // HE_m1.data_size)
        return "✅ Model & HE context initialized successfully!"
    except Exception as e:
        return f"❌ Initialization failed: {str(e)}"

def load_mnist_digits():
    global loaded_data, loaded_labels

    if HE_m1 is None:
        return [], "⚠️ Please initialize model & context first!"

    test_dataset = datasets.MNIST(
        root=os.path.join(root_dir, "Data"),
        train=False,
        transform=TRANSFORM,
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=num_of_data, shuffle=True, drop_last=True
    )
    data, labels = next(iter(test_loader))
    loaded_data = data
    loaded_labels = labels.tolist()

    images = [d.squeeze(0).numpy() for d in data]
    return [(img, f"Label: {label}") for img, label in zip(images, loaded_labels)], "✅ Loaded MNIST digits!"

def encrypt_batch():
    global encrypted_batch

    if loaded_data is None:
        return "⚠️ Load data first!"

    ppData = preprocessing(
        np.array(loaded_data),
        MNIST_Img,
        num_of_data,
        HE_m1.data_size
    )

    encrypted_batch = HE_m1.encrypt(ppData)
    return "✅ Batch encrypted successfully!"

def batch_encrypted_inference():
    global decrypted_batch

    if encrypted_batch is None:
        return "⚠️ Encrypt data first!"

    encrypted_result = HE_m1(encrypted_batch, _time=True)
    decrypted_batch = HE_m1.decrypt(encrypted_result)
    return "✅ Batch inference completed!"

def decrypt_and_compare():
    if decrypted_batch is None:
        return "⚠️ Run batch encrypted inference first!"

    result_images = []
    result_plain = []
    result_he = []
    result_error = []

    correct = 0

    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()

    for i in range(num_of_data):
        origin_results = m1_model(loaded_data)[i].flatten().tolist()
        origin_label = origin_results.index(max(origin_results))

        he_result = -1
        MIN_VALUE = -1e10
        sum_error = 0
        for idx in range(10):
            he_output = decrypted_batch[idx + HE_m1.data_size * i]
            sum_error += np.abs(origin_results[idx] - he_output)
            if MIN_VALUE < he_output:
                MIN_VALUE = he_output
                he_result = idx

        if he_result == origin_label:
            correct += 1

        # Draw
        img = Image.fromarray((loaded_data[i].squeeze().numpy() * 255).astype(np.uint8)).convert("RGB")
        img_resized = img.resize((64, 64))
        draw = ImageDraw.Draw(img_resized)
        draw.text((2, 2), f"HE: {he_result}", fill="green", font=font)
        draw.text((2, 16), f"Plain: {origin_label}", fill="blue", font=font)

        result_images.append(img_resized)
        result_plain.append(f"{origin_label}")
        result_he.append(f"{he_result}")
        result_error.append(f"{sum_error:.6f}")

    accuracy = (correct / num_of_data) * 100
    acc_text = f"✅ Accuracy: {accuracy:.2f}%"

    return result_images, "\n".join(result_plain), "\n".join(result_he), "\n".join(result_error), acc_text

def clear_all():
    global loaded_data, loaded_labels, encrypted_batch, decrypted_batch
    loaded_data = None
    loaded_labels = None
    encrypted_batch = None
    decrypted_batch = None
    return [], "", "", "", "", ""

# ========== Gradio UI ==========
with gr.Blocks(title="Batch Encrypted Inference") as app:
    gr.Markdown("# 🔐 Batch Encrypted Inference on MNIST")

    # 🔹 Step 0: Initialize
    gr.Markdown("## Step 0: 🧠 Initialize Model & HE Context")
    init_btn = gr.Button("🧠 Initialize Model & Keys")
    init_status = gr.Textbox(label="Status", interactive=False)

    # 🔹 Step 1: Load digits
    gr.Markdown("## Step 1: 📥 Load MNIST Digits")
    load_btn = gr.Button("📥 Load MNIST Digits")
    images_gallery = gr.Gallery(label="Loaded Images", show_label=True, columns=5)
    load_status = gr.Textbox(label="Status", interactive=False)

    # 🔹 Step 2: Encrypt Batch
    gr.Markdown("## Step 2: 🔒 Encrypt Batch")
    encrypt_btn = gr.Button("🔒 Encrypt Batch")
    encrypt_status = gr.Textbox(label="Status", interactive=False)

    # 🔹 Step 3: Batch Encrypted Inference
    gr.Markdown("## Step 3: 🚀 Run Batch Encrypted Inference")
    infer_btn = gr.Button("🚀 Run Encrypted Inference")
    infer_status = gr.Textbox(label="Status", interactive=False)

    # 🔹 Step 4: Decrypt & Compare
    gr.Markdown("## Step 4: 🔎 Decrypt & Compare Results")
    compare_btn = gr.Button("🔎 Decrypt & Compare Results")
    result_images = gr.Gallery(label="Results", show_label=True, columns=5)
    result_plain = gr.Textbox(label="Plaintext Results", interactive=False)
    result_he = gr.Textbox(label="HE Results", interactive=False)
    result_error = gr.Textbox(label="Error", interactive=False)
    accuracy_box = gr.Textbox(label="Accuracy", interactive=False)

    # 🔹 Final: Reset
    gr.Markdown("## 🔁 Reset App")
    clear_btn = gr.Button("🧹 Clear Results")

    # Button logic
    init_btn.click(fn=initialize_model_and_keys, outputs=init_status)
    load_btn.click(fn=load_mnist_digits, outputs=[images_gallery, load_status])
    encrypt_btn.click(fn=encrypt_batch, outputs=encrypt_status)
    infer_btn.click(fn=batch_encrypted_inference, outputs=infer_status)
    compare_btn.click(fn=decrypt_and_compare, outputs=[result_images, result_plain, result_he, result_error, accuracy_box])
    clear_btn.click(fn=clear_all, outputs=[
        images_gallery, load_status, encrypt_status, infer_status,
        result_images, result_plain, result_he, result_error, accuracy_box
    ])

# ========== Launch ==========
app.launch()

