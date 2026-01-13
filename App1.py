import streamlit as st

def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n-1)
    
def main():
    st.title("Factorial Calculator")
    num = st.number_input("Enter a number", min_value=0)
    result = factorial(num)
    st.write(f"The factorial of {num} is {result}")

if __name__ == "__main__":
    main()
