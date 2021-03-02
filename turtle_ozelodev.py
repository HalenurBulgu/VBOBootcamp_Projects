
import turtle

screen = turtle.Screen()
t = turtle.Pen()

speed_value=int(input("input your turtle's speed"))

turtle.bgcolor("Black")
color = str(input("choose your best color"))

def make_spiral():
    t=turtle.Turtle()
    t.speed(speed_value)
    for i in range(500):
        t.pencolor(color)
        t.width(i/1000 +1)
        t.forward(i)
        t.left(70)


make_spiral()

screen.exitonclick()
