import time
import json
from datetime import datetime
from collections import defaultdict
import os

class LocalMonitor:
    def __init__(self, log_file="logs/monitoring.json"):
        self.log_file = log_file
        self.requests = []
        os.makedirs("logs", exist_ok=True)
    
    def log_request(self, user, question, sql, duration_ms, status):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "question": question,
            "sql": sql,
            "duration_ms": duration_ms,
            "status": status
        }
        self.requests.append(entry)
        
        if len(self.requests) > 1000:
            self.requests = self.requests[-1000:]
        
        with open(self.log_file, 'w') as f:
            json.dump(self.requests[-100:], f, indent=2)
    
    def get_stats(self):
        if not self.requests:
            return {"message": "No requests yet"}
        
        total = len(self.requests)
        avg_time = sum(r["duration_ms"] for r in self.requests) / total
        errors = sum(1 for r in self.requests if r["status"] != 200)
        
        return {
            "total_requests": total,
            "avg_response_time_ms": round(avg_time, 2),
            "error_rate": round(errors / total * 100, 2),
            "status": "healthy"
        }

monitor = LocalMonitor()