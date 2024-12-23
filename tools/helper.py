import os
import base64
from weasyprint import HTML
import os

def MoveCharts(src_dir="charts", dest_dir="Dumped Charts"):
    """
    Moves files from the 'charts' directory to the 'Dumped Charts' directory,
    renaming files to avoid overwriting existing files in the destination directory.
    """
    import os
    import shutil
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        print(f"Source directory '{src_dir}' does not exist.")
        return

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Scan the source directory for files
    for file_name in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, file_name)

        # Skip directories and handle only files
        if os.path.isfile(src_file_path):
            dest_file_path = os.path.join(dest_dir, file_name)

            # If a file with the same name exists in the destination, rename it
            base_name, ext = os.path.splitext(file_name)
            counter = 1
            while os.path.exists(dest_file_path):
                dest_file_path = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                counter += 1

            # Move the file to the destination
            shutil.move(src_file_path, dest_file_path)
            print(f"Moved '{file_name}' to '{dest_file_path}'.")

def load_images_from_folder(folder_path):
    """
    Load image paths from the specified folder and convert them to base64.

    Args:
        folder_path (str): The path to the folder containing images.

    Returns:
        list: A list of image HTML `<img>` tags with base64-encoded images.
    """
    image_html = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # Ensure valid image formats
            # Get absolute path to the image file
            image_path = os.path.join(folder_path, filename)
            
            # Open the image file and convert it to base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create an HTML img tag with base64 image and reduce size to fit the layout
            img_tag = f'<div style="flex: 1; margin: 10px; text-align: center; width: 45%;">' \
                      f'<img src="data:image/png;base64,{encoded_image}" alt="Chart" ' \
                      f'style="max-width: 90%; height: auto; border: 1px solid #BDC3C7; border-radius: 5px; padding: 10px;"/>' \
                      f'</div>'
            image_html.append(img_tag)
    
    # Group the images in rows of 2 (2 per row)
    grouped_images_html = []
    for i in range(0, len(image_html), 2):
        grouped_images_html.append(
            f'<div style="display: flex; justify-content: space-between; flex-wrap: wrap;">' +
            ''.join(image_html[i:i+2]) +
            '</div>'  # Wrap 2 images per row
        )
    
    return grouped_images_html

def save_to_pdf(markdown_content, file_name="output.pdf", charts_folder="charts"):
    """
    Save Markdown content to a PDF file with enhanced styling and appended images from a folder.

    Args:
        markdown_content (str): The Markdown content to be saved.
        file_name (str): Name of the output PDF file.
        charts_folder (str): Path to the folder containing images to append to the PDF.
    """
    import markdown
    from weasyprint import HTML

    # Convert Markdown to HTML
    html_content = markdown.markdown(
        markdown_content,
        extensions=["markdown.extensions.tables", "markdown.extensions.fenced_code"]
    )

    # KaTeX integration: Add scripts and styles for math rendering
    katex_css = """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.css">
    """
    katex_js = """
    <script src="https://cdn.jsdelivr.net/npm/katex/dist/katex.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/katex/dist/contrib/auto-render.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            renderMathInElement(document.body, {
                delimiters: [
                    {left: "\\[", right: "\\]", display: true},  // Block math
                    {left: "\\(", right: "\\)", display: false}  // Inline math
                ]
            });
        });
    </script>
    """

    # Load images from the charts folder
    images_html = "".join(load_images_from_folder(charts_folder))

    # Final HTML content with KaTeX for math rendering and images
    full_html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        {katex_css}
        <style>
            @page {{
                size: A4;
                margin: 15mm;
            }}
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                color: #2C3E50;
                margin: 0;
            }}
            h1, h2, h3 {{
                color: #34495E;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            h1 {{
                font-size: 24px;
                border-bottom: 2px solid #E74C3C;
                padding-bottom: 3px;
            }}
            h2 {{
                font-size: 20px;
                border-left: 4px solid #3498DB;
                padding-left: 8px;
                color: #34495E;
            }}
            h3 {{
                font-size: 18px;
                color: #16A085;
            }}
            p {{
                margin: 8px 0;
                text-align: justify;
            }}
            ul {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            li {{
                margin-bottom: 5px;
            }}
            strong {{
                font-weight: bold; /* Only bold the text without changing the color */
            }}
            code {{
                background-color: #F8F9F9;
                padding: 1px 3px;
                border-radius: 3px;
                font-family: 'Courier New', Courier, monospace;
            }}
            pre {{
                background-color: #F4F6F6;
                padding: 8px;
                border-left: 3px solid #BDC3C7;
                border-radius: 5px;
                overflow-x: auto;
            }}
            blockquote {{
                margin: 15px 0;
                padding: 10px 15px;
                background-color: #F9F9F9;
                border-left: 4px solid #E67E22;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            table th, table td {{
                border: 1px solid #BDC3C7;
                padding: 6px;
                text-align: left;
            }}
            table th {{
                background-color: #ECF0F1;
                font-weight: bold;
            }}
            footer {{
                text-align: center;
                margin-top: 30px;
                font-size: 12px;
                color: #7F8C8D;
            }}
        </style>
    </head>
    <body>
        {html_content}
        <h2 style="color:#34495E; margin-top: 30px;">Relevant Charts Generated:</h2>
        {images_html}
        {katex_js}
        <footer>Generated by Custom Markdown-to-PDF Converter</footer>
    </body>
    </html>
    """

    # Convert HTML to PDF
    html = HTML(string=full_html_content)
    html.write_pdf(file_name)

    MoveCharts()
    print(f"PDF saved as {file_name}")



import os

def append_to_file(filename, text):
    """
    Appends the given text to a .txt file.
    If the file exists, it is deleted before appending.

    Args:
        filename (str): The name of the .txt file.
        text (str): The text to append.
    """
    try:
        # Open file in append mode (it creates the file if not present)
        with open(filename, 'a') as file:
            file.write(text + '\n')  # Append the text followed by a newline
        print(f"Text appended to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")