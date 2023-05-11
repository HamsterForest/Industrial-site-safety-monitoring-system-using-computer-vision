import tkinter as tk
import cv2

def show_image():
    root.withdraw() # Hide the root window
    img_color = cv2.imread('videos/trafficcone.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('Frame', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    root.deiconify() # Show the root window after cv2 window is closed
    

# Create the main window
root = tk.Tk()
root.title("Monitoring system")
root.geometry("640x400")
root.resizable(False, False)

# Create the buttons
button1 = tk.Button(root, text="접근금지구역 모니터링", command=show_image)
button2 = tk.Button(root, text="작업인원수 모니터링")
button3 = tk.Button(root, text="적치물 제한 구역 모니터링")

# Place the buttons in the window
button1.place(x=50, y=70, width=250, height=50)
button2.place(x=50, y=170, width=250, height=50)
button3.place(x=50, y=270, width=250, height=50)

# Run the GUI
root.mainloop()

