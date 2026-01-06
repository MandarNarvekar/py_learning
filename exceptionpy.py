try:
    a=10
    b=0
    c=a/b
    print(c)
except ZeroDivisionError as e:
    print("Error: Division by zero is not allowed.", e)     
except Exception as e:
    print("An unexpected error occurred:", e)   
finally:
    print("Execution completed.")   