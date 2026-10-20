import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from Models.predict import predict_defect


# --------------------------------------------------
# 1. Configure the Streamlit application
# --------------------------------------------------

st.set_page_config(
    page_title="Industrial Steel Defect Detection",
    page_icon="🔍",
    layout="centered"
)


# --------------------------------------------------
# 2. Display the application title and description
# --------------------------------------------------

st.title("Industrial Steel Defect Detection")

st.write(
    "Upload a steel surface image to identify its defect "
    "category using our trained ResNet18 deep learning model."
)

st.info(
    "The model classifies images into six steel surface "
    "defect categories."
)


# --------------------------------------------------
# 3. Create the image uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Choose a steel surface image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# 4. Display the uploaded image
# --------------------------------------------------

if uploaded_file is not None:

    try:
        # Open the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display the image
        st.image(
            image,
            caption="Uploaded Steel Surface Image",
            use_container_width=True
        )

        # Display image dimensions
        st.write(
            f"Image dimensions: {image.width} × {image.height}"
        )

        # --------------------------------------------------
        # 5. Predict the defect when the button is clicked
        # --------------------------------------------------

        if st.button("Predict Defect", type="primary"):

            temporary_image_path = None

            try:
                with st.spinner("Analyzing the steel surface..."):

                    # Preserve the uploaded file extension
                    suffix = Path(uploaded_file.name).suffix.lower()

                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix,
                        delete=False
                    ) as temporary_file:

                        # Write the uploaded image bytes
                        temporary_file.write(
                            uploaded_file.getvalue()
                        )

                        # Store the temporary file path
                        temporary_image_path = Path(
                            temporary_file.name
                        )

                    # Run the trained model
                    result = predict_defect(
                        temporary_image_path
                    )

                # --------------------------------------------------
                # 6. Display the prediction results
                # --------------------------------------------------

                st.success("Prediction completed successfully!")

                st.subheader("Prediction Results")

                st.write(
                    f"**Predicted defect:** "
                    f"{result['predicted_class'].replace('_', ' ').title()}"
                )

                st.metric(
                    label="Model Confidence",
                    value=f"{result['confidence']:.2f}%"
                )

                # Display all six defect categories
                st.caption(
                    "Available classes: crazing, inclusion, patches, "
                    "pitted surface, rolled-in scale, and scratches."
                )

                st.warning(
                    "Confidence represents the model's estimated "
                    "probability for its predicted class. It does not "
                    "guarantee that the prediction is correct."
                )

            except Exception as error:
                st.error(
                    f"Prediction failed: {error}"
                )

            finally:
                # Delete the temporary image
                if temporary_image_path is not None:
                    temporary_image_path.unlink(
                        missing_ok=True
                    )

    except Exception as error:
        st.error(
            f"Unable to open the uploaded image: {error}"
        )

else:
    st.info(
        "Upload a JPG, JPEG, or PNG image to get started."
    )


# --------------------------------------------------
# 7. Display project information
# --------------------------------------------------

st.divider()

st.caption(
    "Model: ResNet18 | Task: Steel Surface Defect Classification | "
    "Framework: PyTorch"
)