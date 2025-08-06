import discord
from discord.ext import commands
import gdown
import logging
from types import FunctionType
from dotenv import load_dotenv
import os
from pathlib import Path
import re
import asyncio
import tqdm
from utils import run_inference_in_docker


env_path = Path('.') / '.env'

print(f"Loading .env from: {env_path.absolute()}")

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
logger = logging.getLogger('discord')
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

load_dotenv(dotenv_path=env_path)

# debug prints
print(f"Current working directory: {os.getcwd()}")
print(f"Does .env exist? {os.path.exists('.env')}")

token = ""
try:
    with open(env_path) as f:
        for line in f:
            if line.startswith('DISCORD_TOKEN='):
                token = line.split('=', 1)[1].strip()
                # Set environment variable manually
                os.environ['DISCORD_TOKEN'] = token
                break
except Exception as e:
    print(f"Error reading .env file: {e}")

if not token:
    raise ValueError("Could not load Discord token!")

intents = discord.Intents.default()
intents.message_content = True # manually enable the intents that you checked in the UI of developer portal

bot = commands.Bot(command_prefix='/', intents=intents)

@bot.event
async def on_ready():
    print(f"We are ready to go in, {bot.user.name}")

dm_sent_users = set()

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only process DMs (private messages)
    if isinstance(message.channel, discord.DMChannel):
        print(f"Received DM from {message.author.name}: {message.content}")
        await bot.process_commands(message)
    # Ignore messages in guild (server) channels

@bot.command()
async def start(ctx):
    # Only allow /start in DMs
    if isinstance(ctx.channel, discord.DMChannel):
        await ctx.send(
            f"Hello {ctx.author.mention}! \nTo test your model, please use the command ```/evaluate```\nBut before, please read the rules using ```/rules``` command.\n\n"
        )
    else:
        await ctx.send("Please DM me and use `/start` there to begin.")

@bot.command()
async def rules(ctx):
    rules_text = (
        "## Here are the rules for Evaluating your model:\n"
        "1. Your model must be built using PyTorch (mandatory).\n"
        "2. The model should be saved in a public Google Drive link (.pt or .pth), you will be asked to provide its link later on.\n"
        "3. You must upload a Python file containing the `InferenceEngine` class with `preprocess`, `postprocess`, and `run_model` methods defined for your custom use case.\n"
        "4. The evaluation will run for 200 seconds, in an Nvidia A40 GPU so ensure your model can handle this time frame.\n"
        "5. If your model fails to evaluate, you will receive a list of errors and issues. Please try to diagnose and debug them.\n"
        "6. Any attempt to make the system behave unexpectedly will result in immediate disqualification from the competition. Your activity is being logged, and you are fully responsible for all your actions.\n"
        "7. Finally If you are struggling with any issue, do not hesitate to reach out to the mentors.\n"
        "Attached is the InferenceEngine class that you should implement for your model and a sample_inference python file (make sure you keep the same signature of functions)\n\n"
    )
    await ctx.send(rules_text)

    inference_engine_path = "inference_template.py"
    if not os.path.exists(inference_engine_path):
        print("InferenceEngine template file not found.")
        return False
    
    inference_engine_sample = "sample_inference.py"
    if not os.path.exists(inference_engine_sample):
        print("Sample inference is not found")
        return False
    
    try:
        with open(inference_engine_path, 'rb') as file:
            discord_file = discord.File(file, filename="inference_template.py")
            await ctx.send(
                file=discord_file
            )
        with open(inference_engine_sample, 'rb') as file:
            discord_file = discord.File(file, filename="sample_inference.py")
            await ctx.send(
                file=discord_file
            )
        return True
    
    except FileNotFoundError:
        await ctx.send("Results file not found.")
        return False
    except discord.errors.HTTPException as e:
        await ctx.send(f"Error uploading file: File might be too large. {str(e)}")
        return False
    except Exception as e:
        await ctx.send(f"Error uploading results: {str(e)}")
        return False


@bot.command()
async def evaluate(ctx):
    URL_PATTERN = r'(https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*))'

    # 1. Fetching the model url from the user
    await ctx.send(f"Please {ctx.author.mention} provide me with the link to your model (.pth)")
    
    deadline = asyncio.get_event_loop().time() + 200
    
    while True:
        try:
            remaining_time = deadline - asyncio.get_event_loop().time()
            if remaining_time <= 0:
                raise asyncio.TimeoutError()
            
            def check(message):
                return message.author == ctx.author and message.channel == ctx.channel
            
            response = await bot.wait_for('message', check=check, timeout=remaining_time)
            
            # Check if the message contains a valid URL
            if re.match(URL_PATTERN, response.content):
                await ctx.send("Valid URL received!")
                url = response.content.strip()

                # 2. Getting the inference.py file from the user
                user_inference_py = await wait_for_python_file(ctx)
                if not user_inference_py:
                    print("No valid Python file uploaded")
                    return False
                
                # 3. Download the model from Google Drive
                is_valid, msg_or_url = valid_drive_link(url)
                if not is_valid:
                    await ctx.send(f"Invalid model URL: {msg_or_url}")
                    return False
                url = msg_or_url
                print(f"Valid model URL: {url}")
                await ctx.send(f"Downloading the model from the provided URL...")
                download_success = await download_from_drive(ctx, url)
                if not download_success:
                    await ctx.send("Failed to download the model from the provided URL. Please check the link and try again.")
                    return False
                
                await ctx.send(f"Model downloaded successfully!")

                # 4. Run the inference in a docker container

                # run_inference_in_docker(USER_ID):
                await ctx.send(f"Running the inference for your model...")
                execution_output = run_inference_in_docker(ctx.author)
                if "error" in execution_output:
                    await ctx.send(f"Error during inference: {execution_output['error']}")
                    return False
                
                print(f"Execution output: {execution_output}")

                error_file = os.path.join(execution_output["temp_dir"], "error.txt")
                opath = execution_output["predictions_file"]

                if os.path.exists(error_file):
                    with open(error_file, 'r') as ef:
                        error_content = ef.read()
                    await ctx.send(f"Your model failed to evaluate. Here are the errors:\n```\n{error_content}\n```")
                    return False

                try:
                    with open(opath, 'rb') as file:
                        discord_file = discord.File(file, filename="submission.csv")
                    
                        await ctx.send(
                            content="Here is the prediction of your model",
                            file=discord_file
                        )

                    return True
                except FileNotFoundError:
                    await ctx.send("Results file not found.")
                    return False
                except discord.errors.HTTPException as e:
                    await ctx.send(f"Error uploading file: File might be too large. {str(e)}")
                    return False
                except Exception as e:
                    await ctx.send(f"Error uploading results: {str(e)}")
                    return False
                
            else:
                await ctx.send(f"That's not a valid URL. Please provide a valid link. You have {int(remaining_time)} seconds remaining.")
                continue
                
        except asyncio.TimeoutError:
            await ctx.send(f"Sorry, you didn't provide a valid link within the time limit! Try again please :)")
            return None
        except Exception as e:
            await ctx.send(f"Error: {str(e)}")
            return None    
        

####################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################


async def download_from_drive(ctx, url):
    try:
        await ctx.send(f"Downloading the model...")
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{ctx.author}.pth"

        # Start download
        with tqdm.tqdm(desc="Downloading model", unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            success = gdown.download(
                url=url,
                output=str(model_path),
                quiet=False,
                fuzzy=True,
                use_cookies=False
            )

        if success and os.path.exists(model_path):
            print(f"Model downloaded and saved as `{model_path}`.")
            return True

    except Exception as e:
        print(f"Error while downloading model: {str(e)}")
        return None

    

def valid_drive_link(model_link):
    # compare against google drive regex to check whether the model link is valid or not
    if not model_link:
        return False
    
    url = model_link.strip()

    # Check if URL starts with http(s)://
    if not url.startswith(('http://', 'https://')):
        return False, "URL must start with http:// or https://"
    
    # Check if it's a Google Drive domain
    if not any(domain in url for domain in ['drive.google.com', 'docs.google.com']):
        return False, "Not a Google Drive URL"
    
    # Common Drive URL patterns
    drive_patterns = [
        r'https?://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',           # Direct file link
        r'https?://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',         # Open link
        r'https?://drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)',    # Folder link
        r'https?://docs\.google\.com/uc\?id=([a-zA-Z0-9_-]+)'            # Direct download link
    ]
    
    for pattern in drive_patterns:
        if re.match(pattern, url):
            return True, url
            
    return False, "Invalid Google Drive URL format"


# functions to receive preprocessing and postprocessing functionalities by the bot
async def wait_for_python_file(ctx):
    await ctx.send("Please upload a `.py` file containing the InferenceEngine which you have defined.")

    def check(message):
        return (
            message.author == ctx.author and
            message.channel == ctx.channel and
            message.attachments and
            message.attachments[0].filename.endswith(".py")
        )

    try:
        message = await bot.wait_for("message", check=check, timeout=60)
        attachment = message.attachments[0]
        file_path = f"inference/{ctx.author}.py"
        os.makedirs("inference", exist_ok=True)
        await attachment.save(file_path)
        return file_path
    
    except asyncio.TimeoutError:
        await ctx.send("Timeout: No Python file was uploaded.")
        return None



bot.run(token, log_handler=handler, log_level=logging.DEBUG)# is what allows you to access ur developer application