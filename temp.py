import sqlite3
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox

def main():
    #hide the main tkinter window
    root = Tk()
    root.withdraw()

    #ask user to select a file
    file_name = askopenfilename(title="Please select the Assignment5input.txt file")

    #chack if the file is selected
    if not file_name:
        print("No inpout file selected. Exiting......")
        return
    

    #create and initialize the database connection
    conn = sqlite3.connect('temperatures.db')
    cursor = conn.cursor()


    #creating the table with the specified fields
    cursor.execute('''
            CREATE TABLE IF NOT EXISTS temperatures_data (
                ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Day_of_Week TEXT,
                Temperature_value REAL
                )    
            ''') 
    

    #Open the file and read the data
    try:
        with open(file_name, 'r') as file:
            for line in file:
                day, temp = line.strip().split()
                cursor.execute('''
                               INSERT INTO temperatures_data (Day_of_Week, Temperature_value)
                               VALUES (?, ?)
                               ''', (day, float(temp)))
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        conn.close()
        return
    except ValueError:
        print(f"Error: The file {file_name} contains invalid data format")
        conn.close()
        return
    
    #execute the SQL Commands to computer the average Temperature
    cursor.execute('''
                   SELECT AVG(Temperature_value) FROM temperatures_data WHERE Day_of_Week = 'Sunday'
                   ''')
    result_sunday = cursor.fetchone()[0]


    #execute the SQL Commands to computer the average Temperature
    cursor.execute('''
                   SELECT AVG(Temperature_value) FROM temperatures_data WHERE Day_of_Week = 'Thursday'
                   ''')
    result_thursday = cursor.fetchone()[0]

    #print the results
    print(f"Average Temperature on Sunday: {result_sunday:.2f}")
    print(f"Average Temperature on Thursday: {result_thursday:.2f}")
    #show the results in a message box
    messagebox.showinfo("Average Temperatures", f"Average Temperature on Sunday: {result_sunday:.2f}\nAverage Temperature on Thursday: {result_thursday:.2f}")

    #commit changes and close the connection
    conn.commit()
    conn.close()
    #show a message box to indicate the completion of the task
    messagebox.showinfo("Task Completed", "The average tempferatures have been calculated and stored in the database.")

    
if __name__ == "__main__":
        main()
                