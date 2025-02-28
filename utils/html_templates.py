"""
HTML templates for the Streamlit app
"""
import base64
import os

def get_base64_encoded_image(image_path):
    """Load and encode images to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load and encode images at module level
try:
    bot_img = get_base64_encoded_image("images/ideo.png")
    user_img = get_base64_encoded_image("images/user.png")
except Exception as e:
    bot_img = ""
    user_img = ""
    print(f"Error loading chat icons: {str(e)}")

# Load CSS from file
def load_css():
    """Load CSS from file"""
    try:
        # Update path to the new location
        with open("styles/styles.css", "r") as css_file:
            return css_file.read()
    except Exception as e:
        print(f"Error loading CSS: {str(e)}")
        # Fallback CSS if file can't be loaded
        return """
        .chat-message {
            padding: 1.5rem; 
            margin-bottom: 1rem; 
            display: flex;
            position: relative;
        }
        .chat-message.user {
            background-color: white;
        }
        .chat-message.bot {
            background-color: #f7f7f8;
            position: relative;
        }
        .chat-message .avatar {
          width: 20%;
        }
        .chat-message .avatar img {
          max-width: 78px;
          max-height: 78px;
          border-radius: 50%;
          object-fit: cover;
        }
        .chat-message .message {
          width: 80%;
          padding: 0 1.5rem;
          color: #000;
        }
        """

# Load CSS - but don't include it directly in templates
css = load_css()

# Bot message template
bot_template = f"""
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{bot_img}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""

# User message template
user_template = f"""
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_img}" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{{{MSG}}}}</div>
</div>
"""

# Add new debug template
debug_template = """
<div class="debug-info">
    <h4>Debug Information</h4>
    {{DEBUG_INFO}}
</div>
"""

# Function to inject CSS properly
def inject_css():
    """Return CSS wrapped in style tags for proper injection"""
    return f"""
    <style>
    {css}
    .debug-info {{
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }}
    .debug-info pre {{
        white-space: pre-wrap;
        word-wrap: break-word;
    }}
    </style>
    """