package main

import (
	"fmt"
	"net/http"
	"os"
	//"strconv"
	"time"

	"github.com/gin-gonic/gin"
)

var db = make(map[string]string)

type Data struct {
	classification string
	image          [][]bool
}

func flatten(array [][]bool) []byte {
	flattened := []byte{}
	for _, row := range array {
		for _, i := range row {
			fmt.Println("im not tweaking")
			if i {
				flattened = append(flattened, 1)
			} else {
				flattened = append(flattened, 0)

			}
		}
	}

	fmt.Println(flattened)
	return flattened
}

func setupRouter() *gin.Engine {
	// Disable Console Color
	// gin.DisableConsoleColor()
	router := gin.Default()
	router.LoadHTMLGlob("html/*")

	// Ping test
	router.GET("/ping", func(c *gin.Context) {
		c.String(http.StatusOK, "pong")
	})

	// Get user value
	router.GET("/user/:name", func(c *gin.Context) {
		user := c.Params.ByName("name")
		value, ok := db[user]
		if ok {
			c.JSON(http.StatusOK, gin.H{"user": user, "value": value})
		} else {
			c.JSON(http.StatusOK, gin.H{"user": user, "status": "no value"})
		}
	})

	router.GET("/index", func(c *gin.Context) {
		c.HTML(http.StatusOK, "index.html", gin.H{})
	})

	router.POST("/data", func(c *gin.Context) {
		//var data Data
		//if err := c.BindJSON(&data); err != nil {
		//	c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		//	return
		//}
		class := c.PostForm("classification")
		image := c.PostForm("image")
		fmt.Println(class)
		now := time.Now()
		currentTime := now.Format("2006-01-02 15:04:05")
		os.WriteFile("./collected/"+class+"/"+currentTime, []byte(image), 0666)

	})

	return router
}

func main() {
	r := setupRouter()
	// Listen and Server in 0.0.0.0:8080
	r.Run(":8080")
}

/*
	// Authorized group (uses gin.BasicAuth() middleware)
	// Same than:
	// authorized := r.Group("/")
	// authorized.Use(gin.BasicAuth(gin.Credentials{
	//	  "foo":  "bar",
	//	  "manu": "123",
	//}))
	authorized := r.Group("/", gin.BasicAuth(gin.Accounts{
		"foo":  "bar", // user:foo password:bar
		"manu": "123", // user:manu password:123
	}))

	/* example curl for /admin with basicauth header
	   Zm9vOmJhcg== is base64("foo:bar")

		curl -X POST \
	  	http://localhost:8080/admin \
	  	-H 'authorization: Basic Zm9vOmJhcg==' \
	  	-H 'content-type: application/json' \
	  	-d '{"value":"bar"}'

	authorized.POST("admin", func(c *gin.Context) {
		user := c.MustGet(gin.AuthUserKey).(string)

		// Parse JSON
		var json struct {
			Value string `json:"value" binding:"required"`
		}

		if c.Bind(&json) == nil {
			db[user] = json.Value
			c.JSON(http.StatusOK, gin.H{"status": "ok"})
		}
	})*/
