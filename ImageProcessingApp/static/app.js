document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById('image');
    const fileNameDisplay = document.getElementById('file-name');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const imagePreview = document.getElementById('image-preview');

    // Handle file input change event
    fileInput.addEventListener('change', function () {
        const file = fileInput.files[0];

        if (file) {
            // Display file name
            fileNameDisplay.textContent = file.name;

            // Create a preview of the image
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block'; // Show image preview
                imagePreviewContainer.style.display = 'block'; // Show the container
            };
            reader.readAsDataURL(file);
        } else {
            // If no file is selected, reset preview
            fileNameDisplay.textContent = 'No file chosen';
            imagePreview.style.display = 'none';
            imagePreviewContainer.style.display = 'none';
        }
    });
});
