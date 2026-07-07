#=========================================
# Logistic Regression Disease Prediction
#=========================================

rm(list=ls())

#=========================================
# 1. 加载R包
#=========================================

library(readxl)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)

#=========================================
# 2. 读取数据
#=========================================

data <- read_excel("D:/r studio 4.6.0/template-0608 修改版.xlsx")

#=========================================
# 3. 删除无关列
#=========================================

remove_cols <- c(
  "Sample number",
  "Sample name",
  "Other phenotypic information",
  "…(Other phenotypic information)"
)

remove_cols <- intersect(remove_cols,names(data))

data2 <- data[,!(names(data) %in% remove_cols)]

#=========================================
# 4. 数据类型转换
#=========================================

data2$Age <- as.numeric(data2$Age)
data2$Gender <- as.numeric(data2$Gender)
data2$Group <- as.factor(data2$Group)

metabolite_cols <- grep("^Metabolite",names(data2))

data2[,metabolite_cols] <-
  lapply(data2[,metabolite_cols],as.numeric)

#=========================================
# 5. 删除缺失值
#=========================================

data2 <- na.omit(data2)

#=========================================
# 6. 划分训练集/测试集
#=========================================

set.seed(123)

index <- createDataPartition(
  data2$Group,
  p=0.8,
  list=FALSE
)

train <- data2[index,]

test <- data2[-index,]

cat("训练集：",nrow(train),"\n")
cat("测试集：",nrow(test),"\n")

#=========================================
# 7. 建立Logistic Regression
#=========================================

glm_model <- glm(
  Group~.,
  data=train,
  family=binomial()
)

summary(glm_model)

#=========================================
# 8. 预测概率
#=========================================

prob <- predict(
  glm_model,
  newdata=test,
  type="response"
)

pred <- ifelse(prob>0.5,1,0)

pred <- factor(
  pred,
  levels=c(0,1)
)

#=========================================
# 9. 混淆矩阵
#=========================================

cm <- confusionMatrix(
  pred,
  test$Group,
  positive="1"
)

print(cm)

cat("\nAccuracy =",cm$overall["Accuracy"],"\n")

cat("Sensitivity =",cm$byClass["Sensitivity"],"\n")

cat("Specificity =",cm$byClass["Specificity"],"\n")

#=========================================
# 10. ROC
#=========================================

roc_obj <- roc(
  test$Group,
  prob
)

auc_value <- auc(roc_obj)

print(auc_value)

#=========================================
# 11. 保存图片路径
#=========================================

save_path <- "D:/r studio 4.6.0/Logistic_Result/"

dir.create(
  save_path,
  showWarnings=FALSE
)

#=========================================
# 12. ROC图
#=========================================

png(
  paste0(save_path,"ROC.png"),
  width=2000,
  height=1800,
  res=300
)

plot(
  roc_obj,
  col="#D55E00",
  lwd=4,
  main="Logistic Regression ROC Curve"
)

abline(
  a=0,
  b=1,
  lty=2,
  col="grey60"
)

text(
  0.65,
  0.15,
  paste("AUC =",round(auc_value,3)),
  cex=2
)

dev.off()

#=========================================
# 13. 混淆矩阵图片
#=========================================

cm_table <- table(
  True=test$Group,
  Predicted=pred
)

cm_df <- melt(cm_table)

cm_df$True <- factor(
  cm_df$True,
  levels=c("0","1"),
  labels=c("Group 0","Group 1")
)

cm_df$Predicted <- factor(
  cm_df$Predicted,
  levels=c("0","1"),
  labels=c("Group 0","Group 1")
)

p <- ggplot(
  cm_df,
  aes(Predicted,True)
)+
  geom_tile(
    aes(fill=value),
    colour="white"
  )+
  geom_text(
    aes(label=value),
    size=8,
    fontface="bold"
  )+
  scale_fill_gradient(
    low="#EFF3FF",
    high="#2166AC"
  )+
  theme_classic(base_size=18)+
  labs(
    title="Confusion Matrix",
    x="Predicted Class",
    y="True Class"
  )

ggsave(
  paste0(save_path,"Confusion_Matrix.png"),
  p,
  width=6,
  height=5,
  dpi=600
)

#=========================================
# 14. 保存预测结果
#=========================================

result <- data.frame(
  True=test$Group,
  Probability=prob,
  Prediction=pred
)

write.csv(
  result,
  paste0(save_path,"Prediction_Result.csv"),
  row.names=FALSE
)

cat("\n============================\n")
cat("Logistic Regression完成！\n")
cat("AUC =",auc_value,"\n")
cat("============================\n")
