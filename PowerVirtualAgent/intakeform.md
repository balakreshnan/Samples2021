# Power Virtual agent - Onboarding new projects to Data lake

## Sizing and getting basic information to invoke Solution architecting

## Pre requistie

- Azure Account
- Azure Power Platform
- Need dynamics 365 power platform environment to build
- https://web.powerva.microsoft.com/

## Creating a BoT

- Log into https://web.powerva.microsoft.com/
- Go to Topics

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img1.jpg "Service Health")

- Select Greetings
- Select Go to Authoring canvas

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img2.jpg "Service Health")

- Now create a new Topic

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img4.jpg "Service Health")

- Name is DLBot
- Click the DLBot and go to Authoring canvas
- First greetings like how can i help
- Use Show Message to display the greetings
- Next create a new task called Ask a question
- "What is your Storage Size in GB?"
- Select user input

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img5.jpg "Service Health")

- Assign the output to a variable called: storagesize
- Add one more question
- "How much data would process each day in GB?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img6.jpg "Service Health")

- Assign to variable cdcsize
- Add one more question
- "How many data processing job?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img7.jpg "Service Health")

- Assign to variable jobno
- Add one more question
- "How many data processing job?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img7.jpg "Service Health")

- Assign to variable jobno
- Add one more question
- "How much time would all job takes in a day?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img8.jpg "Service Health")

- Assign to variable jobtimeday
- Add one more question
- "How are the job schedule across the day?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img9.jpg "Service Health")

- Assign to variable jobschedule
- Add one more question
- "What is your Application Number?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img10.jpg "Service Health")

- Assign to variable appid
- Add one more question
- "Who is the application owner to contact?"
- select user response

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img11.jpg "Service Health")

- Assign to variable contact
- Now Display the data 
- Add show message

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img12.jpg "Service Health")

## Create a flow to send email

- now lets grab all the data from variables and send an email
- For testing i am using my own personal one
- Click Create actions as next step
- Will navigate you to new screen for flow development
- Here is the full flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img14.jpg "Service Health")

- Create new variables for input

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img15.jpg "Service Health")

- Send Email

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img16.jpg "Service Health")

- Return results

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img17.jpg "Service Health")

- Next go back to bot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img13.jpg "Service Health")

- Send result out

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img18.jpg "Service Health")

- Now test the bot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img19.jpg "Service Health")

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img20.jpg "Service Health")

- here is the output to flow

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img21.jpg "Service Health")

- Here is the output

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/PowerVirtualAgent/images/img22.jpg "Service Health")