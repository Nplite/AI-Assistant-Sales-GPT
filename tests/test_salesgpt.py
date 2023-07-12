import pytest
from langchain.chat_models import ChatOpenAI

from salesgpt.agents import SalesGPT


class TestSalesGPT:
    def test_valid_inference(self, load_env):
        """Test that the agent will start and generate the first utterance."""

        llm = ChatOpenAI(temperature=0.9)

        sales_agent = SalesGPT.from_llm(llm, verbose=False,
                            salesperson_name="VarahiBot",
                            salesperson_role="Sales Representative",
                            company_name="Varahi Technology",
                            company_business='''Varahi Technologies is a dynamic and forward-thinking technology company that has established a 
                            reputation for providing innovative solutions to our clients. We are passionate about technology and 
                            committed to making a difference in the world through our work. Our company is built on a foundation of 
                            strong values and a culture that values collaboration, diversity, and continuous learning. We believe in 
                            empowering our employees to be the best they can be and to make a difference in the world through their 
                            work. At Varahi Technologies, we are committed to excellence, and we are excited about the future. Varahi Technologies' software technologies and innovative services and products:
                            Software Technologies:
                            • Varahi Technologies has expertise in a wide range of software technologies, including Java, .NET, 
                            Python, Ruby on Rails, and more.
                            • We have experience working with various databases and frameworks and can develop custom 
                            software solutions to meet our client's specific needs.
                            • Our team stays current with the latest technologies and trends, constantly learning and exploring 
                            new tools and techniques to deliver cutting-edge solutions.
                            Innovative Services and Products:
                            • We offer a variety of innovative services and products, including custom software development, 
                            mobile app development, cloud computing, e-commerce solutions, and more.
                            • Our team has experience developing complex, mission-critical applications for a variety of industries, 
                            including healthcare, finance, education, and more.
                            • We also offer a range of product development services, including product ideation, design, 
                            development, and marketing, to help startups and entrepreneurs bring their ideas to life.
                            Overall, Varahi Technologies is a full-service technology solutions provider that can help clients with all 
                            aspects of software development and innovation. Whether you need a custom software solution, a mobile 
                            app, or a product development partner, we have the expertise and experience to deliver high-quality results 
                            that exceed your expectations.''')
        
        
        sales_agent.seed_agent()
        sales_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt

        # agent output sample
        sales_agent.step()

        agent_output = sales_agent.conversation_history[-1]
        assert agent_output is not None, "Agent output cannot be None."
        assert isinstance(agent_output, str), "Agent output needs to be of type str"
        assert len(agent_output) > 0, "Length of output needs to be greater than 0."
    
    
    def test_valid_inference_stream(self, load_env):
        """Test that the agent will start and generate the first utterance when streaming."""

        llm = ChatOpenAI(temperature=0.9)
        model_name = 'gpt-3.5-turbo'

        sales_agent = SalesGPT.from_llm(llm, verbose=False,
                            salesperson_name="VarahiBot",
                            salesperson_role="Sales Representative",
                            company_name="Varahi Technology",
                            company_business='''Varahi Technologies is a dynamic and forward-thinking technology company that has established a 
                            reputation for providing innovative solutions to our clients. We are passionate about technology and 
                            committed to making a difference in the world through our work. Our company is built on a foundation of 
                            strong values and a culture that values collaboration, diversity, and continuous learning. We believe in 
                            empowering our employees to be the best they can be and to make a difference in the world through their 
                            work. At Varahi Technologies, we are committed to excellence, and we are excited about the future. Varahi Technologies' software technologies and innovative services and products:
                            Software Technologies:
                            • Varahi Technologies has expertise in a wide range of software technologies, including Java, .NET, 
                            Python, Ruby on Rails, and more.
                            • We have experience working with various databases and frameworks and can develop custom 
                            software solutions to meet our client's specific needs.
                            • Our team stays current with the latest technologies and trends, constantly learning and exploring 
                            new tools and techniques to deliver cutting-edge solutions.
                            Innovative Services and Products:
                            • We offer a variety of innovative services and products, including custom software development, 
                            mobile app development, cloud computing, e-commerce solutions, and more.
                            • Our team has experience developing complex, mission-critical applications for a variety of industries, 
                            including healthcare, finance, education, and more.
                            • We also offer a range of product development services, including product ideation, design, 
                            development, and marketing, to help startups and entrepreneurs bring their ideas to life.
                            Overall, Varahi Technologies is a full-service technology solutions provider that can help clients with all 
                            aspects of software development and innovation. Whether you need a custom software solution, a mobile 
                            app, or a product development partner, we have the expertise and experience to deliver high-quality results 
                            that exceed your expectations.''')
        
        sales_agent.seed_agent()
        sales_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt

        # agent output sample
        stream_generator = sales_agent.step(return_streaming_generator=True, model_name=model_name)
        agent_output=''
        for chunk in stream_generator:
            token = chunk["choices"][0]["delta"].get("content", "")
            agent_output += token

        assert agent_output is not None, "Agent output cannot be None."
        assert isinstance(agent_output, str), "Agent output needs to be of type str"
        assert len(agent_output) > 0, "Length of output needs to be greater than 0."
