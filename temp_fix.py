import sqlite3
conn = sqlite3.connect('rung.db')
conn.execute("UPDATE users SET is_active=0 WHERE email='alice@example.com'")
conn.commit()
print('Пользователь деактивирован!')
conn.close()
