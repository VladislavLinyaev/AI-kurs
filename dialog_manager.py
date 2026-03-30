# dialog_manager.py
from enum import Enum

class DialogState(Enum):
    START = "start"
    WAIT_CITY = "wait_city"
    WAIT_DATE = "wait_date"

class DialogManager:
    def __init__(self):
        self.user_states = {}
        self.user_data = {}
    
    def get_state(self, user_id: int) -> DialogState:
        if user_id in self.user_states:
            return self.user_states[user_id]
        return DialogState.START
    
    def set_state(self, user_id: int, state: DialogState):
        self.user_states[user_id] = state
    
    def get_data(self, user_id: int) -> dict:
        if user_id not in self.user_data:
            self.user_data[user_id] = {}
        return self.user_data[user_id]
    
    def clear_data(self, user_id: int):
        if user_id in self.user_data:
            self.user_data[user_id] = {}
    
    def reset(self, user_id: int):
        self.set_state(user_id, DialogState.START)
        self.clear_data(user_id)

dialog_manager = DialogManager()