import os
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
]

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = (32, 32)
MODEL_ENV_VAR = 'CIFAR10_MODEL_PATH'

PAGE_CSS = """
<style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        border: 1px solid #e5e7eb;
        border-radius: 1.1rem;
        background: linear-gradient(120deg, #f8fafc 0%, #eef6ff 100%);
    }
    .hero .eyebrow {
        color: #2563eb;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 0.35rem 0 0.45rem;
        color: #0f172a;
        font-size: clamp(2rem, 4vw, 3.1rem);
        line-height: 1.05;
    }
    .hero p {
        max-width: 760px;
        margin: 0;
        color: #475569;
        font-size: 1.05rem;
    }
    .prediction-card {
        padding: 1.25rem 1.35rem;
        border: 1px solid #bfdbfe;
        border-radius: 1rem;
        background: #eff6ff;
    }
    .prediction-card .label {
        color: #1d4ed8;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .prediction-card .class-name {
        margin: 0.35rem 0 0.2rem;
        color: #0f172a;
        font-size: 2.1rem;
        font-weight: 750;
    }
    .prediction-card .confidence {
        color: #475569;
        font-size: 1rem;
    }
    .probability-row {
        margin: 0.55rem 0 1rem;
        padding: 0.8rem 0.9rem;
        border: 1px solid #e2e8f0;
        border-radius: 0.8rem;
        background: #ffffff;
    }
    .probability-head {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: #1e293b;
        font-size: 0.98rem;
    }
    .probability-track {
        height: 0.55rem;
        margin-top: 0.5rem;
        overflow: hidden;
        border-radius: 999px;
        background: #e2e8f0;
    }
    .probability-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
    }
    .soft-note {
        padding: 0.9rem 1rem;
        border-left: 4px solid #60a5fa;
        border-radius: 0.45rem;
        color: #475569;
        background: #f8fafc;
    }
</style>
"""


def resolve_model_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def preprocess_image(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    pixels = np.asarray(image)
    background = np.median(pixels.reshape(-1, 3), axis=0).astype(int)
    difference = np.abs(pixels.astype(int) - background).max(axis=2)
    mask = difference > 25
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))

    if rows.size and columns.size:
        image = image.crop(
            (
                int(columns[0]),
                int(rows[0]),
                int(columns[-1] + 1),
                int(rows[-1] + 1),
            )
        )

    side = max(image.size)
    background_color = tuple(int(channel) for channel in background)
    canvas = Image.new('RGB', (side, side), background_color)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    canvas = canvas.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    return np.asarray(canvas, dtype=np.float32)[np.newaxis, ...] / 255.0


@st.cache_resource(show_spinner=False)
def load_model(model_path):
    import tensorflow as tf

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f'Model file not found: {path}')
    return tf.keras.models.load_model(path, compile=False)


def predict_image(model, image):
    image_batch = preprocess_image(image)
    probabilities = np.asarray(model.predict(image_batch, verbose=0)).reshape(-1)
    if probabilities.size != len(CLASS_NAMES):
        raise ValueError(
            f'Model returned {probabilities.size} classes; expected {len(CLASS_NAMES)}.'
        )
    order = np.argsort(probabilities)[::-1]
    return probabilities, order


def render_probability_row(rank, class_index, probability):
    st.markdown(
        f"""
        <div class="probability-row">
            <div class="probability-head">
                <span>{rank}. {CLASS_NAMES[class_index].title()}</span>
                <strong>{probability:.1%}</strong>
            </div>
            <div class="probability-track">
                <div class="probability-fill" style="width: {probability * 100:.1f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title='CIFAR-10 Vision',
        layout='wide',
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    configured_path = os.getenv(MODEL_ENV_VAR)
    default_path = configured_path or str(PROJECT_ROOT / 'cifar10_cnn.keras')

    with st.sidebar:
        st.header('Model')
        st.caption('The app loads the trained Keras model once and reuses it for every upload.')
        model_path_value = st.text_input('Model path', value=default_path)
        model_path = resolve_model_path(model_path_value)
        if model_path.is_file():
            st.success('Model file found')
        else:
            st.warning('Model file not found')
        st.divider()
        st.subheader('How it works')
        st.markdown(
            """
            1. The uploaded image is converted to RGB.
            2. The subject is cropped, padded to a square, and resized to 32 × 32.
            3. Pixel values are normalized to the range 0–1.
            4. The CNN returns a probability for each CIFAR-10 class.
            """
        )
        st.subheader('Classes')
        st.caption(', '.join(name.title() for name in CLASS_NAMES))

    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">Computer vision demo</div>
            <h1>CIFAR-10 Vision</h1>
            <p>Upload an image and see the model prediction, confidence score, and the three most likely classes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        'Upload an image',
        type=['jpg', 'jpeg', 'png', 'webp', 'bmp'],
        accept_multiple_files=False,
        help='Supported formats: JPG, PNG, WEBP, and BMP.',
    )

    if uploaded_file is None:
        st.markdown(
            '<div class="soft-note">Choose an image above to run a prediction.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        uploaded_file.seek(0)
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        st.error('The uploaded file could not be opened as an image. Please try another file.')
        return

    if not model_path.is_file():
        st.error(
            f'No trained model was found at `{model_path}`. Save the notebook model as '
            '`cifar10_cnn.keras`, or enter a different model path in the sidebar.'
        )
        st.code("model.save('cifar10_cnn.keras')", language='python')
        return

    try:
        with st.spinner('Loading the trained model...'):
            model = load_model(str(model_path))
    except ModuleNotFoundError:
        st.error('TensorFlow is not installed in this environment. Install the app requirements and restart Streamlit.')
        return
    except Exception as error:
        st.error(f'The model could not be loaded: {error}')
        return

    try:
        with st.spinner('Analyzing the image...'):
            probabilities, order = predict_image(model, image)
    except Exception as error:
        st.error(f'The prediction could not be completed: {error}')
        return

    top_index = int(order[0])
    top_probability = float(probabilities[top_index])

    image_column, result_column = st.columns([1, 1])
    with image_column:
        st.subheader('Uploaded image')
        st.image(image, use_container_width=True)
        st.caption(uploaded_file.name)

    with result_column:
        st.subheader('Prediction')
        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="label">Top prediction</div>
                <div class="class-name">{CLASS_NAMES[top_index].title()}</div>
                <div class="confidence">Confidence: {top_probability:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('')
        st.caption('The model returns a probability distribution across all ten CIFAR-10 classes.')

    st.subheader('Top 3 predictions')
    for rank, class_index in enumerate(order[:3], start=1):
        render_probability_row(rank, int(class_index), float(probabilities[class_index]))

    with st.expander('View all class probabilities'):
        rows = [
            {
                'Rank': rank,
                'Class': CLASS_NAMES[int(class_index)].title(),
                'Probability': f'{float(probabilities[int(class_index)]):.2%}',
            }
            for rank, class_index in enumerate(order, start=1)
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)

    st.caption('Preprocessing matches the notebook: RGB input, 32 × 32 pixels, and pixel values scaled to 0–1.')


if __name__ == '__main__':
    main()
