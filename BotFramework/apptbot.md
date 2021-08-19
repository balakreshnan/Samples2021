# Appointment Reschedule Bot

## How to build a Conversational AI bot to reschedule appointments

## Prerequistie

- Azure Account
- Download Bot frame composer
- installation location - https://docs.microsoft.com/en-us/composer/introduction?tabs=v2x
- get the latest version
- Need some Azure credit
- Create a resource group
- Need LUIS
- Now it's build it

## Code

- Create a resource group
- Create a new resource for LUIS
- We need this for the bot
- Once you create the LUIS - cognitive services you are good
- now open Bot composer

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img1.jpg "Service Health")

- Now select language based bot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img2.jpg "Service Health")

- Now create a project with name for the bot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img3.jpg "Service Health")

- Make sure the bot Lanugage understanding dispatcher is set to default.

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img4.jpg "Service Health")

- If not connect the luis to the resource created above
- we don't need to create any LUIS project
- If LUIS is not refreshed please save and close and reopen bot composer and the project
- Now we need to create a trigger in the main appointmentbot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img5.jpg "Service Health")

- for trigger phrase

```
- reschedule
- can't make it today
- please reschedule
- can i change my schedule
- want to reschedule
```

- We still haven't got dialog created
- next is create a new dialog called rescheduleDialog
- this is the dialog to use for rescheduling

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img6.jpg "Service Health")

- Once reschdule is triggered we need to create a new dialog to process the inputs
- Create a new diaglog 

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img7.jpg "Service Health")

- Once we collect the data to process the above dialog will close the conversation
- Now start the bot

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img8.jpg "Service Health")

- now lets test the bot in web chat

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/BotFramework/images/img9.jpg "Service Health")

- as the above image the user enters date and we can process further.
- More to come ...