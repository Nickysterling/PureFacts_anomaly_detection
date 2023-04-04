import tkinter as tk

def go_to_page2():
    page1.grid_forget()
    page2.grid(row=0, column=0, sticky="nsew")

def go_to_home():
    page2.grid_forget()
    page1.grid(row=0, column=0, sticky="nsew")

def submit1():
    print("Submit button 1 pressed. Value entered:", input1.get())

def submit2():
    print("Submit button 2 pressed. Value entered:", input2.get())

def submit3():
    print("Submit button 3 pressed. Value entered:", input3.get())

root = tk.Tk()
root.title("2 Page GUI")
root.geometry("800x600")

page1 = tk.Frame(root)
page2 = tk.Frame(root)

# Page 1
label1 = tk.Label(page1, text="Homepage")
label1.pack(pady=10, padx=20)

button1 = tk.Button(page1, text="Go to Page 2", command=go_to_page2)
button1.pack(pady=10, padx=20)

# Page 2
label2 = tk.Label(page2, text="Page 2")
label2.pack(pady=10, padx=20)

button2 = tk.Button(page2, text="Go to Home", command=go_to_home)
button2.pack(pady=10, padx=20)

# Fee inputs
group_details = tk.Label(page2, text="Fee Group Details")
group_details.pack(pady=10, padx=20)

input1 = tk.Entry(page2)
input1.pack()

submit_button1 = tk.Button(page2, text="Submit", command=submit1)
submit_button1.pack()

sch_assign_details = tk.Label(page2, text="Fee Schedule Assignment Details")
sch_assign_details.pack(pady=10, padx=20)

input2 = tk.Entry(page2)
input2.pack()

submit_button2 = tk.Button(page2, text="Submit", command=submit2)
submit_button2.pack()

sch_assign = tk.Label(page2, text="Fee Schedule Assignment")
sch_assign.pack(pady=10, padx=20)

input3 = tk.Entry(page2)
input3.pack()

submit_button3 = tk.Button(page2, text="Submit", command=submit3)
submit_button3.pack()

# Show the first page
page1.grid(row=0, column=0, sticky="nsew")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

root.mainloop()
