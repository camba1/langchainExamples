import gradio


def greet(name, intensity):
    return "Hello " * intensity + name + "!"


demo = gradio.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs="text",
    title="Greetings",
    description="Greet someone",
    article="Greet someone by name",
    allow_flagging="never",
    examples=[
        ["John", 0],
        ["Mary", 1],
        ["Bob", 2]
    ]
)

demo.launch()
