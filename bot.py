from handlers import process_message
from logger import log_message

def main():
    print("Бот запущен. Введите 'выход' для завершения.")
    
    while True:
        user_input = input("Вы: ")
        
        if user_input.lower() in ['выход', 'exit']:
            print("Бот: До свидания!")
            break
            
        response = process_message(user_input)
        print("Бот:", response)
        log_message(user_input, response)

if __name__ == "__main__":
    main()