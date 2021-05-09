# Process PDF and extract Tables

## Using Form Recognizer - Cognitive Services

## Architecture

![alt text](https://github.com/balakreshnan/Samples2021/blob/main/AzureAI/images/aisamples.jpg "Service Health")

## Steps

- Create a Azure Acoount
- Create Storage account
    - Create a pdfinput container
    - Create a pdfoutput container
- Create Form Recognizer
- Create Functions

## Details

- Create the input container
- Make the input container as blob type
- No Cors enabled
- Create output container
- Open Visual Studio Code
- Install Azure Functions Extension
- Install C#
- Create a new Function App
- Select Blob Trigger and provide pdfinput as container name
- Select .net 3.0
- Select the storage account for storage
- Install Azure.AI.FormRecognizer from Nuget
- Now write code to read the blob pdf and extract table information

```
using System;
using System.IO;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Host;
using Microsoft.Extensions.Logging;

using Azure;
using Azure.AI.FormRecognizer;  
using Azure.AI.FormRecognizer.Models;
using Azure.AI.FormRecognizer.Training;

using System.Collections.Generic;
using System.Threading.Tasks;
using System.Text;

namespace formprocessing.Function
{
    public static class BlobTriggerCSharp1
    {
        private static readonly string endpoint = "https://cogacctname.cognitiveservices.azure.com/";
        private static readonly string apiKey = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
        private static readonly AzureKeyCredential credential = new AzureKeyCredential(apiKey);

        [FunctionName("BlobTriggerCSharp1")]
        public async static void Run([BlobTrigger("pdfinputpython/{name}", Connection = "cogstorage_STORAGE")]Stream myBlob, [Blob("pdfoutput/{name}", FileAccess.Write)] Stream outputcsv, string name, ILogger log)
        {
            log.LogInformation($"C# Blob trigger function Processed blob\n Name:{name} \n Size: {myBlob.Length} Bytes");
            var credential = new AzureKeyCredential(apiKey);
            var client = new FormRecognizerClient(new Uri(endpoint), credential);
            string completeuri = "https://cushwakestorage.blob.core.windows.net/pdfinputpython/" + name; 

            Uri formUri = new Uri(completeuri);

            //Response<FormPageCollection> response = client.StartRecognizeContentFromUri(formUri);
            Response<FormPageCollection> response = await client.StartRecognizeContentFromUriAsync(formUri).WaitForCompletionAsync();
            //response.Wait();
            FormPageCollection formPages = response.Value;
            

            foreach (FormPage page in formPages)
            {
                Console.WriteLine($"Form Page {page.PageNumber} has {page.Lines.Count} lines.");

                /*
                for (int i = 0; i < page.Lines.Count; i++)
                {
                    FormLine line = page.Lines[i];
                    Console.WriteLine($"  Line {i} has {line.Words.Count} {(line.Words.Count == 1 ? "word" : "words")}, and text: '{line.Text}'.");

                    if (line.Appearance != null)
                    {
                        // Check the style and style confidence to see if text is handwritten.
                        // Note that value '0.8' is used as an example.
                        if (line.Appearance.Style.Name == TextStyleName.Handwriting && line.Appearance.Style.Confidence > 0.8)
                        {
                            Console.WriteLine("The text is handwritten");
                        }
                    }

                    Console.WriteLine("    Its bounding box is:");
                    Console.WriteLine($"    Upper left => X: {line.BoundingBox[0].X}, Y= {line.BoundingBox[0].Y}");
                    Console.WriteLine($"    Upper right => X: {line.BoundingBox[1].X}, Y= {line.BoundingBox[1].Y}");
                    Console.WriteLine($"    Lower right => X: {line.BoundingBox[2].X}, Y= {line.BoundingBox[2].Y}");
                    Console.WriteLine($"    Lower left => X: {line.BoundingBox[3].X}, Y= {line.BoundingBox[3].Y}");
                }
                */

                for (int i = 0; i < page.Tables.Count; i++)
                {
                    StringBuilder sb = new StringBuilder();
                    FormTable table = page.Tables[i];
                    Console.WriteLine($"  Table {i} has {table.RowCount} rows and {table.ColumnCount} columns.");
                    string line = "";
                    foreach (FormTableCell cell in table.Cells)
                    {
                        if(cell.ColumnIndex >= (table.ColumnCount - 1))
                        {
                            line = line.TrimEnd(',');
                            sb.AppendLine(line);
                            line = "";
                        }
                        //Console.WriteLine($"    Cell ({cell.RowIndex}, {cell.ColumnIndex}) contains text: '{cell.Text}'.");
                        line += cell.Text + ",";
                    }
                    //sb.AppendLine(line);
                    Console.WriteLine("Table Output : " + sb.ToString());
                }

                /*
                for (int i = 0; i < page.SelectionMarks.Count; i++)
                {
                    FormSelectionMark selectionMark = page.SelectionMarks[i];
                    Console.WriteLine($"  Selection Mark {i} is {selectionMark.State}.");
                    Console.WriteLine("    Its bounding box is:");
                    Console.WriteLine($"      Upper left => X: {selectionMark.BoundingBox[0].X}, Y= {selectionMark.BoundingBox[0].Y}");
                    Console.WriteLine($"      Upper right => X: {selectionMark.BoundingBox[1].X}, Y= {selectionMark.BoundingBox[1].Y}");
                    Console.WriteLine($"      Lower right => X: {selectionMark.BoundingBox[2].X}, Y= {selectionMark.BoundingBox[2].Y}");
                    Console.WriteLine($"      Lower left => X: {selectionMark.BoundingBox[3].X}, Y= {selectionMark.BoundingBox[3].Y}");
                }
                */
            }

            

            log.LogInformation($"Completed function Processed blob\n Name:{name} \n Size: {myBlob.Length} Bytes");

        }


        

    }
}
```

- Press F5 to run the code
- Upload a pdf to the pdfinput container
- View the output in Console log.