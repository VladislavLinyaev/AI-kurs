from datetime import datetime

def log_message(user, bot):
   
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open("chat_log.txt", "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] USER: {user}\n")
        f.write(f"[{timestamp}] BOT: {bot}\n")
        f.write("-" * 50 + "\n")  

def read_logs():

    try:
        with open("chat_log.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Файл логов еще не создан."