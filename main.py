import time
from handlers import process_message
from tts_module import speak_async
 
def main():
    voice_mode = False  
 
    print("=" * 45)
    print("        🤖 ЧАТ-БОТ ЗАПУЩЕН")
    print("=" * 45)
    print("  Команды: 'голос' / 'текст' / 'выход'")
    print(f"  Текущий режим: {'🎤 Голосовой' if voice_mode else '⌨️  Текстовый'}")
    print("=" * 45 + "\n")
 
    listen = None 
 
    speak_async("Привет, я робот!")
    while True:
        try:
           
            if voice_mode:
                
                time.sleep(1.5)
                print("🎙️  Говорите...")
                user_input = listen()
                if not user_input or not user_input.strip():
                    print("   (ничего не распознано, попробуйте ещё)")
                    continue
                print(f"Вы (голос): {user_input}")
            else:
                user_input = input("Вы: ").strip()
                if not user_input:
                    continue
 
           
            cmd = user_input.lower().strip()
 
            if cmd in ("выход", "exit", "quit"):
                print("Бот: До свидания!")
                speak_async("До свидания!")
                break
 
            elif cmd in ("голос", "voice"):
                if listen is None:
                    try:
                        from voice import listen as _l
                        listen = _l
                        print("   ✅ Голосовой модуль подключён")
                    except Exception as e:
                        print(f"   ❌ Ошибка: {e}")
                        continue
                voice_mode = True
                msg = "Переключаю на голосовой режим"
                print(f"\n  🎤 {msg}\n")
                speak_async(msg)
                continue
 
            elif cmd in ("текст", "text"):
                voice_mode = False
                msg = "Переключаю на текстовый режим"
                print(f"\n  ⌨️  {msg}\n")
                speak_async(msg)
                continue
 
          
            response = process_message(user_input)
            print(f"Бот: {response}\n")
            speech_thread = speak_async(response)

            if voice_mode and speech_thread:
                
                speech_thread.join()
 
        except KeyboardInterrupt:
            print("\nБот: До свидания!")
            speak_async("До свидания!")
            break
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            continue
 
if __name__ == "__main__":
    main()